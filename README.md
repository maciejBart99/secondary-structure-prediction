# Protein structure predication in an explainable way
Here a description of main functionalities and classes is provided:

## Core
The core folder contains all classes that form a core of this predication system.
### TestManager
This class is responsible for running model tests, it can be used as follows:
```python
from core.loader import *
from models.model import Model
from extensions.basic_output import BasicOutput
from models.model_wrapper import ModelWrapper
from core.test_manager import *

output = BasicOutput()
# to translate q8 to q3
class_adapter = BasicQ3Adapter()
# (model path = empty for train, model instance, classification mode)
test_model = ModelDescriptor('trained/model_9.pth', Model(), ClassificationMode.Q8)
# (model, output, mode, predictions output)
test_config = TestConfig(test_model, output, mode=ClassificationMode.Q8, dump_predictions='pred.npy')
t_dl, test_dl = data_provider.get_data()
# (batch size train, batch size test, train data set, test data set)
data_provider = DataProvider(128, 512, 'data/train.npz', 'data/v.npz', adapter)
test_manager = TestManager(test_config, test_dl, class_adapter)
res = test_manager.test()

# output structure
class TestResult:
    class TestRecord:

        def __init__(self, _id: int, acc: float, _len: int, seq_str: str, pred: str, target: str,
                     possibility: List[int]):
            self.id = _id
            self.acc = acc
            self.len = _len
            self.seq_str = seq_str
            self.pred = pred
            self.target = target
            self.possibility = possibility

    def __init__(self, acc: float, loss: float, records: List[TestRecord]):
        self.acc = acc
        self.loss = loss
        self.records = records
```

### TrainManager
This class is responsible fot the training process, it can be used as follows:
```python
from core.loader import *
from extensions.basic_output import BasicOutput
from core.train_manager import *
from models.model import Model

# (batch size train, batch size test, train data set, test data set)
data_provider = DataProvider(128, 512, 'data/train.npz', 'data/test.npz')
# (model path = empty for train, model instance, classification mode)
model = ModelDescriptor('', Model(), ClassificationMode.Q8)
output = BasicOutput()
# (epochs, output stream, model description, dest folder)
config = TrainConfig(10, output, model, 'trained')
train_manager = TrainManager(config, data_provider)
train_manager.train()
```

### Explain manager
It is a wrapper around Captum integrated gradient computing api. It can be used as follows:
from core.exaplain_manager import *

```python
from core.loader import *
from models.model import Model
from models.model_wrapper import ModelWrapper
from core.exaplain_manager import *

# (batch size train, batch size test, train data set, test data set)
data_provider = DataProvider(128, 512, 'data/train.npz', 'data/test.npz')
# (model description, model wrapper class)
explain_config = ExplainConfig(ModelDescriptor('trained/model_0.pth', Model(), ClassificationMode.Q8), ModelWrapper)
t_dl, test_dl = data_provider.get_data()
explain_manager = ExplainManager(explain_config, test_dl)

# (sequnce, class number, target position)
explain_manager.explain(0, 0, 0)
# a structure is returned
@dataclass
class ExplainResult:
    seq_from: int
    seq_to: int
    data: np.ndarray
```

## Models
Here models are defined, each model has a wrapper class which is used by ExplainManager

## Extensions

### BasicAdapter
This class contains logic which is necessary to encode and decode sequence position description vectors in a physical-property space

```python
from core.loader import *
from extensions.basic_output import BasicOutput
from core.train_manager import *
from models.model import Model

from extensions.basic_adapter import BasicAdapter

to_encode = np.array(best_values[:21])
# (feature matrix, boolean - if we want to decode from feature space)
adapter = BasicAdapter(to_encode, True)
data_provider = DataProvider(128, 512, 'data/train.npz', 'data/test.npz', adapter)
model = ModelDescriptor('', Model(), ClassificationMode.Q8)
output = BasicOutput()
config = TrainConfig(10, output, model, 'trained')
train_manager = TrainManager(config, data_provider)
train_manager.train()
```


### BasicQ3Adapter
This class can transform Q8 target classes to Q3
