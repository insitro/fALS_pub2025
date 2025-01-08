## Running anchor phenotype stats

Statistics for comparing anchor phenotypes (TDP-43 ratio and STMN2 intensity) are computed from well level aggregated metrics.  See paper for description of linear models and paired density matching.

### Running
The output is already available in s3, but you may reproduce by running the main redun task.  From the base of the repo:
```bash
>>> pixi run redun run -c src/fals/scripts/.redun run anchor_penotype_stats.py main --output-path local/path/
```
