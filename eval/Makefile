
EXT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../ext)

.PHONY: all
all: rbp_eval gdeval.pl

gdeval.pl: $(EXT_DIR)/trec-web-2013/src/eval/gdeval.pl
	cp $^ $@

rbp_eval: $(EXT_DIR)/rbp_eval-0.2/rbp_eval/rbp_eval
	cp $^ $@

$(EXT_DIR)/rbp_eval-0.2/rbp_eval/rbp_eval: $(EXT_DIR)/rbp_eval-0.2/Makefile
	make -C $(EXT_DIR)/rbp_eval-0.2

$(EXT_DIR)/rbp_eval-0.2/Makefile:
	cd $(EXT_DIR) \
		&& tar xf rbp_eval-0.2.tar.gz \
		&& cd rbp_eval-0.2 \
		&& ./configure 
