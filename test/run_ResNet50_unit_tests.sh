#!/bin/bash

# Run unit tests for ops for ResNet50

cd ..
# In ngraph-bridge/

# Copy the nnp library to build_cmake/artifacts/lib required for ./gtest_ngtf
LIB=build_cmake/venv-tf-py3/lib/python3.6/site-packages/ngraph_bridge/libnnp_backend.so 
if [ ! -f "$LIB" ]; then
    echo "NNP backend library $LIB does not exist"
    exit
fi
cp LIB build_cmake/artifacts/lib

echo "**************************************************************************"
echo "Run C++ tests"
echo "**************************************************************************"
pushd build_cmake/test
# In ngraph-bridge/build_cmake/test
./gtest_ngtf --gtest_filter="ArrayOps.Tile:MathOps.Add:NNOps.BiasAddGrad:ArrayOps.ExpandDims:VariableTest.*:-*6"
cpp_status = $?

if [$cpp_status!=0]; then
    echo "CPP Tests failed"
    exit
fi


echo "**************************************************************************"
echo "Run Python tests"
echo "**************************************************************************"
pushd python
# In ngraph-bridge/build_cmake/test/python
# install pytest
pip install -U pytest
pytest test_cast.py test_sparse_softmax_cross_entropy_with_logits.py test_l2loss.py test_relugrad.py test_pad.py
pytest_status = $?

if [$pytest_status!=0]; then
    echo "Python Tests failed"
    exit
fi



echo "**************************************************************************"
echo "Run Python BFloat tests"
echo "**************************************************************************"
pushd bfloat16
# In ngraph-bridge/build_cmake/test/python/bfloat16
pytest test_conv2dbackpropfilter_nchw.py test_conv2dbackpropfilter_nhwc.py test_conv2d.py test_fusedbatchnorm_training_nchw.py test_fusedbatchnorm_training_nhwc.py test_maxpoolbackprop_nhwc.py test_maxpoolbackprop_nchw.py
pytest_bfloat_status = $?
if [$pytest_bfloat_status!=0]; then
    echo "Python Bfloat Tests failed"
    exit
fi

