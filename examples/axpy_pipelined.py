# ==============================================================================
#  Copyright 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
os.environ['NGRAPH_TF_BACKEND'] = "INTERPRETER"
#os.environ['NGRAPH_TF_USE_PREFETCH'] = "1"
import ngraph_bridge

import sys


def build_simple_model(input_array):
    # Convert the numpy array to TF Tensor
    input = tf.cast(input_array, tf.float32)

    # Define the Ops
    mul = tf.compat.v1.math.multiply(input_array, 5)
    add = tf.compat.v1.math.add(mul, 10)
    output = add
    return output


def build_data_pipeline(input_array, map_function, batch_size):
    dataset = (tf.data.Dataset.from_tensor_slices(
        (tf.constant(input_array)
        )).map(map_function).batch(batch_size).prefetch(1))

    iterator = dataset.make_initializable_iterator()
    data_to_be_prefetched_and_used = iterator.get_next()

    return data_to_be_prefetched_and_used, iterator


if __name__ == '__main__':
    num_iter = 3
    input_array = list(range(1, num_iter+1))
    multiplier = 10
    for i in range(1, num_iter+1):
        input_array[i - 1] = input_array[i - 1] * i * multiplier
    map_function = lambda x: x * multiplier
    batch_size = 1
    pipeline, iterator = build_data_pipeline(input_array, map_function,
                                             batch_size)
    model = build_simple_model(pipeline)

    with tf.Session() as sess:
        # Initialize the globals and the dataset
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)

        for i in range(1, num_iter+1):
            # Expected value is:
            expected_output = ((input_array[i - 1] * multiplier) * 5) + 10

            # Run one iteration
            output = sess.run(model)

            # Results?
            print("Iteration:", i, " Input: ", input_array[i - 1], " Output: ",
                  output[0], " Expected: ", expected_output)
            sys.stdout.flush()



'''
Hang logs:

[PREFETCH] COMPUTE: Creating the shared object to signal prefetching
Iteration: 1  Input:  10  Output:  510  Expected:  510
[PREFETCH] COMPUTE: DEPTH: 0 skip count; 0
Iteration: 2  Input:  40  Output:  4510  Expected:  2010
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 1
Iteration: 3  Input:  90  Output:  8010  Expected:  4510
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 2
Iteration: 4  Input:  160  Output:  12510  Expected:  8010
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 3
Iteration: 5  Input:  250  Output:  18010  Expected:  12510
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 4
Iteration: 6  Input:  360  Output:  24510  Expected:  18010
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 5
Iteration: 7  Input:  490  Output:  32010  Expected:  24510
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 6
Iteration: 8  Input:  640  Output:  40510  Expected:  32010
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 7
'''

'''

Hang: 1 2 3 ...
Non hang: 1 3 2 ...

// before `GetNextIoTensorsReadyForDeviceExecution` call a `prefetch_dataset_op` butted in (GetNextIoTensorsForDeviceTransfer + AddNextIoTensorsReadyForDeviceExecution)
// so `m_ng_2_tf` is under contention. But it is a threadsafe queue??

// [Check] DEPTH == skipcount. not entering the if (skip_count >= prefetch_buffer_depth)?



More detailed hang logs:

[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[sarkars][compute] created NGraphPrefetchSharedResouce
[sarkars][compute] added NGraphPrefetchSharedResouce to resource manager
[PREFETCH] COMPUTE: Creating the shared object to signal prefetching
Iteration: 1  Input:  10  Output:  510  Expected:  510 <<<<<<<< OK


[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null  <<<<<<<< NOT OK
[PREFETCH] COMPUTE: DEPTH: 0 skip count; 0 <<<<<<


[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_dataset_op] called GetNextIoTensorsForDeviceTransfer
[sarkars][prefetch_dataset_op] called write
[sarkars][prefetch_dataset_op] called AddNextIoTensorsReadyForDeviceExecution


[sarkars][compute] called GetNextIoTensorsReadyForDeviceExecution
Iteration: 2  Input:  40  Output:  4510  Expected:  2010
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 1
-------------------------------------------------------------------------
More detailed non hang logs:

[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[sarkars][compute] created NGraphPrefetchSharedResouce
[sarkars][queue] Add: 1
[sarkars][compute] added NGraphPrefetchSharedResouce to resource manager
[PREFETCH] COMPUTE: Creating the shared object to signal prefetching
Iteration: 1  Input:  10  Output:  510  Expected:  510  <<<<<<<< OK


[sarkars][prefetch_dataset_op] called Lookup  <<<<<<<<  NOT OK
[sarkars][queue] nextavailable: 0
[sarkars][prefetch_dataset_op] called GetNextIoTensorsForDeviceTransfer
[sarkars][prefetch_dataset_op] called write
[sarkars][prefetch_dataset_op] called AddNextIoTensorsReadyForDeviceExecution


[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 0  <<<<<<<<
[sarkars][queue] nextavailable: 0


Iteration: 2  Input:  40  Output:  2010  Expected:  2010
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 1
[sarkars][compute] called GetNextIoTensorsReadyForDeviceExecution
[sarkars][queue] Add: 1
Iteration: 3  Input:  90  Output:  4510  Expected:  4510
'''





'''
Hang:

[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[sarkars][compute] created NGraphPrefetchSharedResouce
[sarkars][queue] Add: 1
[sarkars][compute] added NGraphPrefetchSharedResouce to resource manager
[PREFETCH] COMPUTE: Creating the shared object to signal prefetching
Iteration: 1  Input:  10  Output:  510  Expected:  510

[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 0 skip count; 0

[sarkars][prefetch_dataset_op] called Lookup
[sarkars][queue] nextavailable: 0
[sarkars][prefetch_dataset_op] called GetNextIoTensorsForDeviceTransfer
[sarkars][prefetch_dataset_op] called write
[sarkars][queue] Add: 1
[sarkars][prefetch_dataset_op] called AddNextIoTensorsReadyForDeviceExecution


[sarkars][queue] nextavailable: 0
[sarkars][compute] called GetNextIoTensorsReadyForDeviceExecution
[sarkars][queue] Add: 1
Iteration: 2  Input:  40  Output:  4510  Expected:  2010
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 1 <<<<<<<

non-hang 

[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[sarkars][compute] created NGraphPrefetchSharedResouce
[sarkars][queue] Add: 1
[sarkars][compute] added NGraphPrefetchSharedResouce to resource manager
[PREFETCH] COMPUTE: Creating the shared object to signal prefetching
Iteration: 1  Input:  10  Output:  510  Expected:  510

[sarkars][prefetch_dataset_op] called Lookup
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 0  <<<<<<<
Iteration: 2  Input:  40  Output:  2010  Expected:  2010
[sarkars][queue] nextavailable: 0
[sarkars][prefetch_dataset_op] called GetNextIoTensorsForDeviceTransfer
[sarkars][prefetch_dataset_op] called write
[sarkars][queue] Add: 1
[sarkars][prefetch_dataset_op] called AddNextIoTensorsReadyForDeviceExecution

[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 1
[sarkars][queue] nextavailable: 0
[sarkars][compute] called GetNextIoTensorsReadyForDeviceExecution
[sarkars][queue] Add: 1
Iteration: 3  Input:  90  Output:  4510  Expected:  4510
'''



'''
hang logs: 


[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[sarkars][compute] created NGraphPrefetchSharedResouce
[sarkars][queue] Add: 1
[sarkars][compute] added NGraphPrefetchSharedResouce to resource manager
[PREFETCH] COMPUTE: Creating the shared object to signal prefetching
Iteration: 1  Input:  10  Output:  510  Expected:  510
[sarkars][prefetch_dataset_op] called Lookup

[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 0 skip count; 0
[sarkars][queue] entered GetNextAvailable. Preparing to wait // waits ~1 sec here. as the sleep time of prefetcher is around 1s


[sarkars] entered GetNextIoTensorsForDeviceTransfer
[sarkars][queue] Done waiting in GetNextAvailable // How is there no print from "add" between "Preparing to wait" and "Done waiting". Ans.. maybe they are different queues... 


[sarkars][queue] nextavailable: 0
[sarkars][prefetch_dataset_op] called GetNextIoTensorsForDeviceTransfer
[sarkars][prefetch_dataset_op] called write
[sarkars][queue] Add: 1
[sarkars][prefetch_dataset_op] called AddNextIoTensorsReadyForDeviceExecution

[sarkars][queue] Done waiting in GetNextAvailable

[sarkars][queue] nextavailable: 0
[sarkars][compute] called GetNextIoTensorsReadyForDeviceExecution
[sarkars][queue] Add: 1
Iteration: 2  Input:  40  Output:  4510  Expected:  2010
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 1
[sarkars][queue] entered GetNextAvailable. Preparing to wait


non hang:

[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[sarkars][compute] created NGraphPrefetchSharedResouce
[sarkars][queue] Add: 1
[sarkars][compute] added NGraphPrefetchSharedResouce to resource manager
[PREFETCH] COMPUTE: Creating the shared object to signal prefetching
Iteration: 1  Input:  10  Output:  510  Expected:  510
[sarkars][prefetch_dataset_op] called Lookup

[sarkars] entered GetNextIoTensorsForDeviceTransfer
[sarkars][queue] Done waiting in GetNextAvailable
[sarkars][queue] nextavailable: 0
[sarkars][prefetch_dataset_op] called GetNextIoTensorsForDeviceTransfer
[sarkars][prefetch_dataset_op] called write
[sarkars][queue] Add: 1
[sarkars][prefetch_dataset_op] called AddNextIoTensorsReadyForDeviceExecution
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 0
[sarkars] SKIPPING. Looks bad
Iteration: 2  Input:  40  Output:  2010  Expected:  2010
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 1
[sarkars][queue] Done waiting in GetNextAvailable
[sarkars][queue] nextavailable: 0
[sarkars][compute] called GetNextIoTensorsReadyForDeviceExecution
[sarkars][queue] Add: 1
Iteration: 3  Input:  90  Output:  4510  Expected:  4510

'''


'''
hang
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[sarkars][compute] created NGraphPrefetchSharedResouce
[sarkars][prefetch_shared_data] Add to m_tf_2_ng
[sarkars][queue] Add: 1
[sarkars][compute] added NGraphPrefetchSharedResouce to resource manager
[PREFETCH] COMPUTE: Creating the shared object to signal prefetching
Iteration: 1  Input:  10  Output:  510  Expected:  510
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 0 skip count; 0
[sarkars][prefetch_shared_data] get from m_ng_2_tf
[sarkars][queue] entered GetNextAvailable. Preparing to wait
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_shared_data] get from m_tf_2_ng
[sarkars][queue] Done waiting in GetNextAvailable: m_queue.size() = 1
[sarkars][queue] nextavailable: 0
[sarkars][prefetch_dataset_op] called GetNextIoTensorsForDeviceTransfer
[sarkars][prefetch_dataset_op] called write
[sarkars][prefetch_shared_data] Add to m_ng_2_tf
[sarkars][queue] Add: 1
[sarkars][prefetch_dataset_op] called AddNextIoTensorsReadyForDeviceExecution
[sarkars][queue] Done waiting in GetNextAvailable: m_queue.size() = 1
[sarkars][queue] nextavailable: 0
[sarkars][compute] called GetNextIoTensorsReadyForDeviceExecution
[sarkars][prefetch_shared_data] Add to m_tf_2_ng
[sarkars][queue] Add: 1
Iteration: 2  Input:  40  Output:  4510  Expected:  2010
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 1
[sarkars][prefetch_shared_data] get from m_ng_2_tf
[sarkars][queue] entered GetNextAvailable. Preparing to wait



non-hang
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[sarkars][compute] created NGraphPrefetchSharedResouce
[sarkars][prefetch_shared_data] Add to m_tf_2_ng
[sarkars][queue] Add: 1
[sarkars][compute] added NGraphPrefetchSharedResouce to resource manager
[PREFETCH] COMPUTE: Creating the shared object to signal prefetching
Iteration: 1  Input:  10  Output:  510  Expected:  510
[sarkars][prefetch_dataset_op] called Lookup
[sarkars][prefetch_shared_data] get from m_tf_2_ng
[sarkars][queue] Done waiting in GetNextAvailable: m_queue.size() = 1
[sarkars][queue] nextavailable: 0
[sarkars][prefetch_dataset_op] called GetNextIoTensorsForDeviceTransfer
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; [sarkars][prefetch_dataset_op] called write
[sarkars][prefetch_shared_data] Add to m_ng_2_tf
0[sarkars][queue] Add: 1

[sarkars][prefetch_dataset_op] called AddNextIoTensorsReadyForDeviceExecution
Iteration: 2  Input:  40  Output:  2010  Expected:  2010
[sarkars][compute] NGRAPH_TF_USE_PREFETCH not null
[PREFETCH] COMPUTE: DEPTH: 1 skip count; 1
[sarkars][prefetch_shared_data] get from m_ng_2_tf
[sarkars][queue] Done waiting in GetNextAvailable: m_queue.size() = 1
[sarkars][queue] nextavailable: 0
[sarkars][compute] called GetNextIoTensorsReadyForDeviceExecution
[sarkars][prefetch_shared_data] Add to m_tf_2_ng
[sarkars][queue] Add: 1
Iteration: 3  Input:  90  Output:  4510  Expected:  4510
'''

# generally it flows: Add to m_tf_2_ng[enc],                          get from m_tf_2_ng[prefetch], Add to m_ng_2_tf[prefetch], get from m_ng_2_tf[enc], Add to m_tf_2_ng[enc]
# Problem if:         Add to m_tf_2_ng[enc], get from m_ng_2_tf[enc], get from m_tf_2_ng[prefetch], Add to m_ng_2_tf[prefetch],                          Add to m_tf_2_ng[enc], get from m_ng_2_tf[prefetch]