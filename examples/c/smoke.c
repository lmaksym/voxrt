/* PLAN 5.1 smoke test: link against libprx and transcribe a WAV file.
   build: see scripts/c_smoke.sh */
#include <stdio.h>
#include <stdlib.h>
#include "prx.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: smoke MODEL_DIR FILE.wav\n"); return 2; }
    PrxTranscriber *t = prx_new(argv[1]);
    if (!t) { fprintf(stderr, "load failed: %s\n", prx_last_error()); return 1; }
    PrxTranscript *r = prx_transcribe_file(t, argv[2]);
    if (!r) { fprintf(stderr, "transcribe failed: %s\n", prx_last_error()); prx_free(t); return 1; }
    printf("prx %s · %.1f s · %zu segments\n%s\n", prx_version(), r->duration, r->n_segments, r->text);
    for (size_t i = 0; i < r->n_segments && i < 2; i++) {
        const PrxSegment *s = &r->segments[i];
        printf("[%.2f-%.2f] %zu words, first: %s (%.2f-%.2f)\n", s->start, s->end, s->n_words,
               s->n_words ? s->words[0].word : "", s->n_words ? s->words[0].start : 0.0, s->n_words ? s->words[0].end : 0.0);
    }
    prx_transcript_free(r);
    prx_free(t);
    return 0;
}
