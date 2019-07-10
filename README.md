# Joint Optimization of Cascade Ranking Models

Implementation of the cascade models used in the paper _Joint Optimization of
Cascade Ranking Models_, WSDM 19.

>L. Gallagher and R-C. Chen and R. Blanco and J. S. Culpepper. 2019. Joint
>Optimization of Cascade Ranking Models. In Proc. WSDM. 15-23. DOI:
>https://doi.org/10.1145/3289600.3290986

## Reproduce Instructions
1. Make sure you have installed the dependencies:

    * g++ or clang++
    * cmake 2.8+
    * Python 3.3+
    * MSLR-WEB10K
    * Yahoo LTR Set 1

2. Clone the repo:

    ```sh
    $ git clone https://github.com/rmit-ir/joint-cascade-ranking
    $ cd joint-cascade-ranking
    $ git submodule update --init --recursive
    ```

3. Build/Install Dependencies

    Build CEGB (note CEGB was added to [LightGBM core][lgbm] after the
    publication of this work):

    ```sh
    $ cd ext/cegb
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ cd ../../..
    ```

    Build evaluation tools:

    ```sh
    $ make -C eval
    ```

    Install Python dependencies:

    ```sh
    $ pip install -r requirements.txt
    $ pip install -e ext/cegb/python-package
    ```

4. Configure Datasets

    Create a `local.mk` file with the path to MSLR and Yahoo datasets:

    ```sh
    $ >local.mk
    $ echo "YAHOO_PATH=/path/to/yahoo" >> local.mk
    $ echo "MSLR_PATH=/path/to/mslr" >> local.mk
    ```

    Link to datasets and create qrels:

    ```sh
    $ make -C exp/yahoo
    $ make -C exp/mslr
    ```

5. Run baselines

    Yahoo:

    ```sh
    $ ./exp/yahoo/baseline-gbrt.sh
    $ ./exp/yahoo/baseline-cegb.sh
    ```

    MSLR:

    ```sh
    $ ./exp/mslr/baseline-gbrt.sh
    $ ./exp/mslr/baseline-cegb.sh
    ```

6. Run joint cascade

    ```sh
    $ ./exp/yahoo/reproduce.sh
    $ ./exp/mslr/reproduce.sh
    ```

[lgbm]: https://github.com/microsoft/LightGBM
