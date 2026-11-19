PY ?= python

.PHONY: install data explore train evaluate profit readme all test app

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

data:        ## download (with checksum) and load into DuckDB, build the split
	$(PY) scripts/01_build_db.py

explore:     ## SQL exploration, randomisation check, ATE
	$(PY) scripts/02_explore.py

train:       ## fit all learners for both outcomes
	$(PY) scripts/03_train.py

evaluate:    ## select on validation, evaluate once on test
	$(PY) scripts/04_evaluate.py

profit:      ## profit curves, policy choice, sensitivity
	$(PY) scripts/05_profit.py

readme:      ## fill README numbers from results/*.json
	$(PY) scripts/render_readme.py

all: data explore train evaluate profit readme

test:
	$(PY) -m pytest -q

app:
	streamlit run app/streamlit_app.py
