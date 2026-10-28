# Candidate A native check

This helper uses `candidate-a-native-v1`. It does not recover the old PR266 parser.

The Linux build runs two exact native tests in separate fresh targets. It then runs 38 fresh production children: two initial calls, two warmup pairs, and 16 fixed AB/BA sample pairs. It checks all 37 later proofs against the complete first proof. The timer contains only the child `output()` call.

The control saves the three exact best files. Other modules use current source paths. Correctness views record actual SMS group values in four fixed shapes. Performance views have no fixture calls. `views.py` rebuilds each expected view from the frozen source for the offline parser.

The current external C source sets LOGQ=48. This C parameter is separate from the commitment Q=40772829184. The helper reads the current stamp and binds actual archives and objects for the parent production build and all three fresh private targets. It binds all seven parent key images and each copied image.

`frozen-source.tsv` binds all source inputs. It omits mutable native archives/object directories and compiler outputs. The package manifest records supplied native artifacts. The runtime records bind the actual native bytes used after the build.

Run the new parser schema tests:

```sh
python3 -B -m unittest discover -s tests -p test_validate.py -v
```

Freeze source inputs after review:

```sh
python3 -B freeze.py --solution .. --protected-root /path/to/trusted/source --context-out /path/to/frozen-context.json
```

The protected scripts do not expose seeds or per-proof PASS fields. Each job evidence JSON must bind `job_id`, `conclusion`, `source_commit`, `source_tree`, `records_sha256`, `score_sha256`, `solution_sha256`, and `protected_sha256` from the actual job. A successful unchanged protected job writes the five-run score only after all five proofs pass. Do not add fields to the protected result.

After request creation, fix an external identity JSON before you read either job result or log. Bind the frozen context SHA-256, exact submitted source commit and tree, and the positive integer official and companion job IDs. The two `jobs` entries use `canonical-1` and `canonical-2`. Keep this record separate from the source payload. The parser requires exact identity matches.

After both fixed canonical jobs finish:

```sh
python3 -B validate.py --solution .. --context /path/to/frozen-context.json --identity /path/to/identity.json \
  --job canonical-1 /path/to/records-1.jsonl /path/to/score-1.json /path/to/job-1.json \
  --job canonical-2 /path/to/records-2.jsonl /path/to/score-2.json /path/to/job-2.json
```

The parser checks eight finite gates. It derives signed deltas from raw integer child durations. Both jobs must have a positive median and mean, at least 12 of 16 positive deltas, and positive AB and BA medians. No floor, tolerance, extra round, or data-based repeat applies.

This Mac cannot run Linux AVX-512 native tests. A parser test, source view, object file, or failed link is not a native correctness or performance PASS.
