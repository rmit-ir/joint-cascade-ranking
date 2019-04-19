-include ../../local.mk
ifneq ($(MAKECMDGLOBALS),clean)
ifndef YAHOO_PATH
$(error Required variable `YAHOO_PATH` not defined. See README.md)
endif
endif

.PHONY: all
all: data qrels

.PHONY: data
data: set1.train.txt set1.valid.txt set1.test.txt

set1.%.txt: $(YAHOO_PATH)/set1.%.txt
	ln -s $^ .

.PHONY: qrels
qrels: set1.test.qrels

set1.test.qrels: set1.test.txt
	awk '{curr = $$2; if (last != curr) { count = 1; last = curr; } print $$2, "0", $$2 "." count++, $$1}' $< | sed 's/qid://g' > $@

.PHONY: clean
clean:
	$(RM) set?.test.qrels set?.{test,train,valid}.txt
