.PHONY: release-check dataset-fixture-check materialize-local-artifacts paper-figures paper-render

release-check:
	python src/release_checks.py

dataset-fixture-check:
	python src/release_checks.py

materialize-local-artifacts:
	python src/materialize_release_artifacts.py --from-local-archive --all

paper-figures:
	python src/generate_scale250_paper_figures.py

paper-render:
	pandoc manuscript/paper.md --from markdown+tex_math_dollars --standalone --resource-path=manuscript -o manuscript/paper.html
	pandoc manuscript/paper.md --from markdown+tex_math_dollars --standalone --resource-path=manuscript -o manuscript/paper.tex
