-include ../../local.mk
ifneq ($(MAKECMDGLOBALS),clean)
ifndef MSLR_PATH
$(error Required variable `MSLR_PATH` not defined. See README.md)
endif
endif

.PHONY: all
all: data qrels

.PHONY: data
data: MSLR-WEB10K qrels

MSLR-WEB10K: $(MSLR_PATH)/MSLR-WEB10K
	ln -s $^ .

.PHONY: qrels
qrels: fold1.test.qrels fold2.test.qrels fold3.test.qrels fold4.test.qrels fold5.test.qrels
	cat $^ > all.test.qrels

fold1.test.qrels: MSLR-WEB10K/Fold1/test.txt
	$(shell pwd)/../../eval/svm2qrel.sh $< > $@
fold2.test.qrels: MSLR-WEB10K/Fold2/test.txt
	$(shell pwd)/../../eval/svm2qrel.sh $< > $@
fold3.test.qrels: MSLR-WEB10K/Fold3/test.txt
	$(shell pwd)/../../eval/svm2qrel.sh $< > $@
fold4.test.qrels: MSLR-WEB10K/Fold4/test.txt
	$(shell pwd)/../../eval/svm2qrel.sh $< > $@
fold5.test.qrels: MSLR-WEB10K/Fold5/test.txt
	$(shell pwd)/../../eval/svm2qrel.sh $< > $@

.PHONY: clean
clean:
	$(RM) -r MSLR-WEB10K *.qrels
