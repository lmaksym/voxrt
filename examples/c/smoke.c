/* PLAN 5.1 smoke test: link against libvoxrt and transcribe a WAV file.
   build: see scripts/c_smoke.sh */
#include <stdio.h>
#include <stdlib.h>
#include "voxrt.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: smoke MODEL_DIR FILE.wav\n"); return 2; }
    VoxrtTranscriber *t = voxrt_new(argv[1]);
    if (!t) { fprintf(stderr, "load failed: %s\n", voxrt_last_error()); return 1; }
    VoxrtTranscript *r = voxrt_transcribe_file(t, argv[2]);
    if (!r) { fprintf(stderr, "transcribe failed: %s\n", voxrt_last_error()); voxrt_free(t); return 1; }
    printf("voxrt %s · %.1f s · %zu segments\n%s\n", voxrt_version(), r->duration, r->n_segments, r->text);
    for (size_t i = 0; i < r->n_segments && i < 2; i++) {
        const VoxrtSegment *s = &r->segments[i];
        printf("[%.2f-%.2f] %zu words, first: %s (%.2f-%.2f)\n", s->start, s->end, s->n_words,
               s->n_words ? s->words[0].word : "", s->n_words ? s->words[0].start : 0.0, s->n_words ? s->words[0].end : 0.0);
    }
    voxrt_transcript_free(r);
    voxrt_free(t);
    return 0;
}
