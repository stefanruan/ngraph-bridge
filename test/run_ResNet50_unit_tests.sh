#!/bin/bash

check_status() {
    if [ $1 -eq 0 ]; then
    echo "**************************************************************************"
    echo "$2 Tests Passed"
    echo "**************************************************************************"
    else
    echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    echo "$2 Tests failed"
    echo "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    exit
    fi
}


# Run unit tests for ops for ResNet50
cd ..
# In ngraph-bridge/

echo "**************************************************************************"
echo "Run C++ tests"
echo "**************************************************************************"
pushd build_cmake/test
# In ngraph-bridge/build_cmake/test
./gtest_ngtf --gtest_filter="ArrayOps.Tile:MathOps.Add:NNOps.BiasAddGrad:ArrayOps.ExpandDims:VariableTest.*:-*6"
cpp_status=$?
check_status $cpp_status "C++"


echo "**************************************************************************"
echo "Run Python tests"
echo "**************************************************************************"
pushd python
# In ngraph-bridge/build_cmake/test/python
# install pytest
pip install -U pytest
pytest test_cast.py test_sparse_softmax_cross_entropy_with_logits.py test_l2loss.py test_relugrad.py test_pad.py
pytest_status=$?
check_status $pytest_status "Python"



echo "**************************************************************************"
echo "Run Python bFloat16 tests"
echo "**************************************************************************"
pushd bfloat16
# In ngraph-bridge/build_cmake/test/python/bfloat16
pytest test_conv2dbackpropfilter_nchw.py test_conv2dbackpropfilter_nhwc.py test_conv2d.py test_fusedbatchnorm_training_nchw.py test_fusedbatchnorm_training_nhwc.py test_maxpoolbackprop_nhwc.py test_maxpoolbackprop_nchw.py
pytest_bfloat_status=$?
check_status $pytest_bfloat_status "Python bFloat16"

