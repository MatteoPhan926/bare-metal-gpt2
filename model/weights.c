// weights.c — loader for the fp32 raw weight file. Order MUST match tools/export_gpt2.py.
#include "weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPT2_MAGIC   0x47505432   /* 'GPT2' */
#define GPT2_VERSION 1

// walk a cursor through the loaded body, handing out sub-arrays in export order
static float *take(float **cur, size_t count) { float *p = *cur; *cur += count; return p; }

int gpt2_load_weights(const char *path, GPT2Weights *w) {
    memset(w, 0, sizeof(*w));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[weights] cannot open %s\n", path); return 1; }

    int hdr[16];
    if (fread(hdr, sizeof(int), 16, f) != 16) { fprintf(stderr, "[weights] short header\n"); fclose(f); return 2; }
    if (hdr[0] != GPT2_MAGIC)   { fprintf(stderr, "[weights] bad magic 0x%08x\n", hdr[0]); fclose(f); return 3; }
    if (hdr[1] != GPT2_VERSION) { fprintf(stderr, "[weights] bad version %d\n", hdr[1]); fclose(f); return 3; }
    // hdr: [magic,version,n_layer,n_head,n_embd,vocab,n_ctx,ffn,qkv,dtype,...]
    if (hdr[2]!=GPT2_N_LAYER || hdr[3]!=GPT2_N_HEAD || hdr[4]!=GPT2_N_EMBD ||
        hdr[5]!=GPT2_VOCAB   || hdr[6]!=GPT2_N_CTX  || hdr[7]!=GPT2_FFN_DIM|| hdr[8]!=GPT2_QKV_DIM) {
        fprintf(stderr, "[weights] dims mismatch vs config.h "
                "(file: L%d H%d E%d V%d C%d F%d Q%d)\n", hdr[2],hdr[3],hdr[4],hdr[5],hdr[6],hdr[7],hdr[8]);
        fclose(f); return 4;
    }
    if (hdr[9] != 0) { fprintf(stderr, "[weights] dtype != fp32 (got %d)\n", hdr[9]); fclose(f); return 5; }

    size_t n = (size_t)GPT2_TOTAL_PARAMS;
    float *body = (float*)malloc(n * sizeof(float));
    if (!body) { fprintf(stderr, "[weights] OOM (%zu floats)\n", n); fclose(f); return 6; }
    size_t got = fread(body, sizeof(float), n, f);
    // ensure the file has EXACTLY the body we expect (not short, not trailing garbage)
    long extra = 0; { long cur = ftell(f); fseek(f, 0, SEEK_END); extra = ftell(f) - cur; }
    fclose(f);
    if (got != n) { fprintf(stderr, "[weights] body short: got %zu / %zu floats\n", got, n); free(body); return 7; }
    if (extra != 0) { fprintf(stderr, "[weights] %ld trailing bytes (layout drift)\n", extra); free(body); return 8; }

    w->data = body; w->n_floats = n;
    float *c = body;
    const int E = GPT2_N_EMBD, V = GPT2_VOCAB, CX = GPT2_N_CTX, Q = GPT2_QKV_DIM, FF = GPT2_FFN_DIM;
    w->wte = take(&c, (size_t)V*E);
    w->wpe = take(&c, (size_t)CX*E);
    for (int L = 0; L < GPT2_N_LAYER; L++) {
        GPT2Layer *ly = &w->layers[L];
        ly->ln1_g = take(&c, E);            ly->ln1_b = take(&c, E);
        ly->qkv_w = take(&c, (size_t)Q*E);  ly->qkv_b = take(&c, Q);
        ly->attn_proj_w = take(&c, (size_t)E*E); ly->attn_proj_b = take(&c, E);
        ly->ln2_g = take(&c, E);            ly->ln2_b = take(&c, E);
        ly->fc_w  = take(&c, (size_t)FF*E); ly->fc_b  = take(&c, FF);
        ly->proj_w= take(&c, (size_t)E*FF); ly->proj_b= take(&c, E);
    }
    w->lnf_g = take(&c, E); w->lnf_b = take(&c, E);

    size_t used = (size_t)(c - body);
    if (used != n) { fprintf(stderr, "[weights] cursor %zu != %zu (layout bug)\n", used, n); free(body); return 9; }
    fprintf(stderr, "[weights] loaded %s : %zu floats (%.1f MB) OK\n", path, n, n*4.0/1e6);
    return 0;
}

void gpt2_free_weights(GPT2Weights *w) { if (w && w->data) { free(w->data); w->data = NULL; } }
