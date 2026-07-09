// correctness.c — STAGE 0 gate harness (QUALITY_GATES §1) for the pure-C fp32 backend.
//
// Runs the CPU forward against the HF fp32 oracle (refdumps/fp32/, ground truth) and applies:
//   (a) per-layer relative error   <= 1e-4  (bug localizer: embed, block_0..11, final_ln)
//   (b) greedy token match (N=128), TEACHER-FORCED so each position is judged on the reference's
//       own trajectory (no divergence cascade); margin rule: a mismatch is a BUG unless the
//       reference top-2 margin at that step < 0.05.  Plus a free-running greedy sanity.
//   (c) distribution agreement over 512 eval positions: top-1 >= 99% AND max KL(ref||ours) < 0.02.
// Then a CPU baseline timing section (prefill latency @P, no-KV decode tok/s) — informational,
// clearly labeled; the rigorous BENCH_PROTOCOL harness is GPU-side (later stages).
//
// Token IDs come from refdumps/meta.json (pre-tokenized HF IDs) so tokenizer coverage never
// confounds kernel correctness (BENCHMARKS.md note). Run from repo root.
//   usage: correctness.exe [all|quick] [weights.bin] [refdir] [meta.json]

#include "forward_cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
static double wtime(void) { return omp_get_wtime(); }
#else
#include <time.h>
static double wtime(void) { return (double)clock() / CLOCKS_PER_SEC; }
#endif

#define E  GPT2_N_EMBD
#define V  GPT2_VOCAB
#define NL GPT2_N_LAYER

// ---------------- small IO / math helpers ----------------
static char *read_text(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[gate] cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char*)malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "[gate] short read %s\n", path); exit(1); }
    buf[n] = 0; fclose(f); return buf;
}
static float *load_f32(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[gate] cannot open %s\n", path); exit(1); }
    float *p = (float*)malloc(n * sizeof(float));
    if (fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "[gate] short read %s (want %zu f32)\n", path, n); exit(1); }
    fclose(f); return p;
}
static int32_t *load_i32(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[gate] cannot open %s\n", path); exit(1); }
    int32_t *p = (int32_t*)malloc(n * sizeof(int32_t));
    if (fread(p, sizeof(int32_t), n, f) != n) { fprintf(stderr, "[gate] short read %s\n", path); exit(1); }
    fclose(f); return p;
}
// extract the JSON integer array following "key" (quotes included in key). returns count.
static int parse_int_array(const char *buf, const char *key, int *out, int maxn) {
    const char *p = strstr(buf, key);
    if (!p) { fprintf(stderr, "[gate] key %s not found in meta.json\n", key); exit(1); }
    p = strchr(p, '[');
    if (!p) { fprintf(stderr, "[gate] no '[' after %s\n", key); exit(1); }
    p++;
    int n = 0;
    while (*p && *p != ']') {
        while (*p && *p != ']' && *p != '-' && (*p < '0' || *p > '9')) p++;
        if (!*p || *p == ']') break;
        char *end; long v = strtol(p, &end, 10);
        if (end == p) { p++; continue; }
        if (n < maxn) out[n] = (int)v;
        n++; p = end;
    }
    return n;
}
static double rel_err(const float *a, const float *b, size_t n) {  // max|a-b| / (max|b| + 1e-9)
    double md = 0.0, mr = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]); if (d > md) md = d;
        double r = fabs((double)b[i]);                if (r > mr) mr = r;
    }
    return md / (mr + 1e-9);
}
static int argmax(const float *v, int n) { int m = 0; for (int i = 1; i < n; i++) if (v[i] > v[m]) m = i; return m; }
// KL(softmax(ref) || softmax(ours)), stable via log-sum-exp.
static double kl_row(const float *ref, const float *ours, int n) {
    double mr = -1e300, mo = -1e300;
    for (int i = 0; i < n; i++) { if (ref[i] > mr) mr = ref[i]; if (ours[i] > mo) mo = ours[i]; }
    double sr = 0.0, so = 0.0;
    for (int i = 0; i < n; i++) { sr += exp((double)ref[i] - mr); so += exp((double)ours[i] - mo); }
    double lsr = mr + log(sr), lso = mo + log(so);
    double kl = 0.0;
    for (int i = 0; i < n; i++) {
        double lpr = (double)ref[i]  - lsr;
        double lpo = (double)ours[i] - lso;
        kl += exp(lpr) * (lpr - lpo);
    }
    return kl;
}
static int cmp_d(const void *a, const void *b) { double x = *(const double*)a, y = *(const double*)b; return (x>y)-(x<y); }
static double median_d(double *v, int n) { qsort(v, n, sizeof(double), cmp_d); return n%2 ? v[n/2] : 0.5*(v[n/2-1]+v[n/2]); }

int main(int argc, char **argv) {
    const char *mode  = (argc > 1) ? argv[1] : "all";
    const char *wpath = (argc > 2) ? argv[2] : "weights/gpt2_124m_fp32.bin";
    const char *refdir= (argc > 3) ? argv[3] : "refdumps/fp32";
    const char *meta  = (argc > 4) ? argv[4] : "refdumps/meta.json";
    int quick = (strcmp(mode, "quick") == 0);
    char path[512];

    printf("==== STAGE 0 correctness gate (pure-C fp32 vs HF fp32 oracle) ====\n");
#ifdef _OPENMP
    printf("build: OpenMP ON, %d threads | fp32 storage, double accumulation | /fp:precise\n\n", omp_get_max_threads());
#else
    printf("build: OpenMP OFF (single thread) | fp32 storage, double accumulation\n\n");
#endif

    GPT2Weights w;
    if (gpt2_load_weights(wpath, &w)) { fprintf(stderr, "weight load failed\n"); return 1; }

    // ---- token IDs from meta.json ----
    char *mbuf = read_text(meta);
    static int prompt_ids[GPT2_N_CTX], eval_ids[GPT2_N_CTX];
    int Tp = parse_int_array(mbuf, "\"prompt_ids\"", prompt_ids, GPT2_N_CTX);
    int Ne = parse_int_array(mbuf, "\"eval_ids\"",   eval_ids,   GPT2_N_CTX);
    free(mbuf);
    printf("[meta] prompt=%d tokens  eval=%d tokens\n\n", Tp, Ne);

    int passA = 1, passB = 1, passC = 1;

    // =================== GATE (a): per-layer relative error ===================
    printf("---- gate (a): per-layer rel_err  (threshold fp32 <= 1e-4) ----\n");
    {
        GPT2Caps c; memset(&c, 0, sizeof(c));
        c.embed    = (float*)malloc((size_t)Tp * E * sizeof(float));
        c.blocks   = (float*)malloc((size_t)NL * Tp * E * sizeof(float));
        c.final_ln = (float*)malloc((size_t)Tp * E * sizeof(float));
        gpt2_forward_cpu(&w, prompt_ids, Tp, &c);

        const double THR = 1e-4;
        // embed
        snprintf(path, sizeof(path), "%s/embed.bin", refdir);
        { float *r = load_f32(path, (size_t)Tp*E); double e = rel_err(c.embed, r, (size_t)Tp*E);
          printf("  %-10s rel_err = %.3e  %s\n", "embed", e, e<=THR?"OK":"** FAIL **"); if(e>THR)passA=0; free(r); }
        // blocks
        for (int L = 0; L < NL; L++) {
            snprintf(path, sizeof(path), "%s/block_%d.bin", refdir, L);
            float *r = load_f32(path, (size_t)Tp*E);
            double e = rel_err(c.blocks + (size_t)L*Tp*E, r, (size_t)Tp*E);
            char nm[16]; snprintf(nm, sizeof(nm), "block_%d", L);
            printf("  %-10s rel_err = %.3e  %s\n", nm, e, e<=THR?"OK":"** FAIL (bug localized to this layer) **");
            if (e > THR) passA = 0;
            free(r);
        }
        // final_ln
        snprintf(path, sizeof(path), "%s/final_ln.bin", refdir);
        { float *r = load_f32(path, (size_t)Tp*E); double e = rel_err(c.final_ln, r, (size_t)Tp*E);
          printf("  %-10s rel_err = %.3e  %s\n", "final_ln", e, e<=THR?"OK":"** FAIL **"); if(e>THR)passA=0; free(r); }
        // localization diagnostic (does NOT alter the locked gate above): if the last-layer tensors
        // fail, test the "oracle dumped POST-ln_f" hypothesis — my ln_f(block11) vs oracle block_11.bin.
        if (!passA) {
            snprintf(path, sizeof(path), "%s/block_11.bin", refdir);
            float *b11 = load_f32(path, (size_t)Tp*E);
            double e = rel_err(c.final_ln, b11, (size_t)Tp*E);   // my post-ln_f hidden vs oracle's block_11
            printf("  [diag] my ln_f(block_11) vs oracle block_11.bin = %.3e %s\n",
                   e, e<=THR ? "-> block_11.bin IS post-ln_f (oracle mislabel; my block_11+ln_f are correct)"
                             : "-> genuine block_11 bug");
            printf("  [diag] final_ln.bin = ln_f(block_11.bin) (double-ln_f artifact; see scratchpad/verify_oracle.py proof)\n");
            free(b11);
        }
        free(c.embed); free(c.blocks); free(c.final_ln);
    }
    printf("  => gate (a): %s\n\n", passA ? "PASS" : "FAIL");

    if (quick) { printf("[quick mode] skipping gates (b),(c) and timing.\n"); gpt2_free_weights(&w); return passA?0:1; }

    // =================== GATE (b): greedy token match (teacher-forced) ===================
    printf("---- gate (b): greedy token match, N=128  (teacher-forced; margin<0.05 tolerates a flip) ----\n");
    int Ng;
    {
        snprintf(path, sizeof(path), "%s/greedy_ids.bin", refdir);
        FILE *f = fopen(path, "rb"); fseek(f,0,SEEK_END); Ng=(int)(ftell(f)/4); fclose(f);
        int32_t *gids = load_i32(path, Ng);
        snprintf(path, sizeof(path), "%s/greedy_margin.bin", refdir);
        float   *gmar = load_f32(path, Ng);

        // teacher-forced sequence: prompt + gids[0..Ng-2]; predictions at pos (Tp-1 .. Tp-2+Ng)
        int slen = Tp + Ng - 1;
        static int seq[GPT2_N_CTX];
        memcpy(seq, prompt_ids, (size_t)Tp*sizeof(int));
        for (int t = 0; t < Ng-1; t++) seq[Tp+t] = gids[t];

        GPT2Caps c; memset(&c,0,sizeof(c));
        c.logits = (float*)malloc((size_t)slen * V * sizeof(float)); c.logits_all = 1;
        gpt2_forward_cpu(&w, seq, slen, &c);

        int match=0, tol=0, bug=0;
        for (int t = 0; t < Ng; t++) {
            int pred = argmax(c.logits + (size_t)(Tp-1+t)*V, V);
            if (pred == gids[t]) { match++; }
            else if (gmar[t] < 0.05f) { tol++;
                printf("    pos %3d: pred %d != ref %d  (ref margin %.4f < 0.05 -> tolerated near-tie)\n", t, pred, gids[t], gmar[t]);
            } else { bug++; passB=0;
                printf("    pos %3d: pred %d != ref %d  (ref margin %.4f >= 0.05 -> ** BUG **)\n", t, pred, gids[t], gmar[t]);
            }
        }
        printf("  teacher-forced: %d/%d match, %d tolerated near-tie, %d bug\n", match, Ng, tol, bug);
        free(c.logits); free(gids); free(gmar);
    }
    printf("  => gate (b): %s\n\n", passB ? "PASS" : "FAIL");

    // =================== GATE (c): distribution agreement over eval window ===================
    printf("---- gate (c): eval-window distribution agreement  (top-1 >= 99%%, max KL < 0.02) ----\n");
    double prefill512_ms = 0.0;
    {
        GPT2Caps c; memset(&c,0,sizeof(c));
        c.logits = (float*)malloc((size_t)Ne * V * sizeof(float)); c.logits_all = 1;
        double t0 = wtime();
        gpt2_forward_cpu(&w, eval_ids, Ne, &c);
        prefill512_ms = (wtime()-t0)*1e3;

        snprintf(path, sizeof(path), "%s/eval_logits.bin", refdir);
        float *r = load_f32(path, (size_t)Ne*V);
        int agree = 0; double maxkl = 0.0;
        for (int p = 0; p < Ne; p++) {
            const float *ro = r + (size_t)p*V, *oo = c.logits + (size_t)p*V;
            if (argmax(ro, V) == argmax(oo, V)) agree++;
            double kl = kl_row(ro, oo, V); if (kl > maxkl) maxkl = kl;
        }
        double agree_pct = 100.0*agree/Ne;
        printf("  top-1 agreement = %d/%d (%.2f%%)   max KL(ref||ours) = %.3e\n", agree, Ne, agree_pct, maxkl);
        if (agree_pct < 99.0 || maxkl >= 0.02) passC = 0;
        free(c.logits); free(r);
    }
    printf("  => gate (c): %s\n\n", passC ? "PASS" : "FAIL");

    // =================== CPU baseline timing (informational) ===================
    printf("---- CPU baseline (informational; not the GPU BENCH_PROTOCOL harness) ----\n");
    {
        // prefill latency @ P: full forward, last-position logits only (what prefill actually needs)
        int Ps[2] = {128, 512}, Rs[2] = {10, 5};   // fewer runs at the slower P (CPU is slow — informational)
        float *lg = (float*)malloc((size_t)V*sizeof(float));
        for (int pi = 0; pi < 2; pi++) {
            int P = Ps[pi], R = Rs[pi];
            GPT2Caps c; memset(&c,0,sizeof(c)); c.logits = lg; c.logits_all = 0;
            gpt2_forward_cpu(&w, eval_ids, P, &c); gpt2_forward_cpu(&w, eval_ids, P, &c);  // warmup 2
            double *s = (double*)malloc(R*sizeof(double));
            for (int r = 0; r < R; r++) { double t0=wtime(); gpt2_forward_cpu(&w, eval_ids, P, &c); s[r]=(wtime()-t0)*1e3; }
            double med = median_d(s, R);           // median_d sorts s ascending -> s[0]=min, s[R-1]=max
            printf("  prefill @P=%-3d : %.1f ms median (min %.1f, max %.1f, N=%d)\n", P, med, s[0], s[R-1], R);
            free(s);
        }
        // no-KV decode: free-running greedy from the prompt (recomputes all past K/V each step)
        snprintf(path, sizeof(path), "%s/greedy_ids.bin", refdir);
        FILE *f=fopen(path,"rb"); fseek(f,0,SEEK_END); int Ng2=(int)(ftell(f)/4); fclose(f);
        int32_t *gids = load_i32(path, Ng2);
        static int seq[GPT2_N_CTX]; memcpy(seq, prompt_ids, (size_t)Tp*sizeof(int)); int slen=Tp;
        double *st = (double*)malloc(Ng2*sizeof(double));
        int lead=0, diverged=-1;
        GPT2Caps c; memset(&c,0,sizeof(c)); c.logits=lg; c.logits_all=0;
        for (int sidx=0; sidx<Ng2; sidx++) {
            double t0=wtime(); gpt2_forward_cpu(&w, seq, slen, &c); st[sidx]=(wtime()-t0)*1e3;
            int tok = argmax(lg, V);
            if (diverged<0) { if (tok==gids[sidx]) lead=sidx+1; else diverged=sidx; }
            seq[slen++] = tok;
        }
        // median per-token over steps 5..Ng2-1 (discard cold), note ctx range (no-KV => grows with ctx)
        int lo=5, cnt=Ng2-lo; double *tail=(double*)malloc(cnt*sizeof(double));
        for (int i=0;i<cnt;i++) tail[i]=st[lo+i];
        double med = median_d(tail, cnt);          // sorts tail; min/max = fastest/slowest step (ctx-dependent)
        printf("  no-KV decode  : %.1f ms/tok median (min %.1f, max %.1f, N=%d) over steps %d..%d, ctx %d->%d -> %.1f tok/s\n",
               med, tail[0], tail[cnt-1], cnt, lo, Ng2-1, Tp+lo, Tp+Ng2-1, 1000.0/med);
        printf("                  (no-KV: per-token time scales ~linearly with ctx -> wide spread is expected; stage-5 KV win = no-KV vs KV under this protocol)\n");
        printf("  free-run greedy vs reference: %d/%d leading tokens match%s\n",
               lead, Ng2, diverged<0 ? " (full)" : "");
        if (diverged>=0) printf("    (first free-run divergence at step %d — expected if a near-tie flipped; see gate (b) verdict)\n", diverged);
        free(gids); free(st); free(tail); free(lg);
        (void)prefill512_ms;
    }
    printf("\n");

    // =================== verdict ===================
    int ok = passA && passB && passC;
    printf("==== STAGE 0 VERDICT: gate(a)=%s gate(b)=%s gate(c)=%s -> %s ====\n",
           passA?"PASS":"FAIL", passB?"PASS":"FAIL", passC?"PASS":"FAIL", ok?"ALL PASS":"FAILED");
    gpt2_free_weights(&w);
    return ok ? 0 : 1;
}
