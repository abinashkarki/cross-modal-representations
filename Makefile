.PHONY: release-check dataset-fixture-check materialize-local-artifacts paper-figures paper-render paper-tmlr

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
	pandoc manuscript/paper.md --from markdown+tex_math_dollars --standalone --resource-path=manuscript \
		--template=manuscript/arxiv.template.tex \
		--lua-filter=manuscript/promote-headings.lua \
		-o manuscript/paper.tex

paper-tmlr:
	pandoc manuscript/tmlr/paper-tmlr.md \
		--from markdown+tex_math_dollars+raw_tex \
		--standalone \
		--resource-path=manuscript/tmlr \
		--template=manuscript/tmlr/tmlr.template.tex \
		--lua-filter=manuscript/promote-headings.lua \
		-o manuscript/tmlr/paper-tmlr.tex
