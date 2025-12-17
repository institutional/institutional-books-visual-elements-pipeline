from .ib_volume import IBVolume
from .pipeline_run import PipelineRun
from .pipeline_batch import PipelineBatch
from .pipeline_batch_item import PipelineBatchItem
from .detection import Detection
from .classification import Classification
from .image_embedding import ImageEmbedding
from .image_hash import ImageHash
from .caption import Caption
from .deduped_hash import DedupedHash
from .deduped_embedding import DedupedEmbedding

"""
TODO: 

@Jimmy-Mendez Thank you very much for this, this is really great work and an important project milestone. 

Here is some high level feedback.

From what I have seen, everything should work as expected and was implemented with care. Because this is a complex pipeline, the more clarity and uniformity we have across the codebase, the better. This will help us troubleshoot issues as we run the pipeline at scale, or more easily understand potential issues with the data itself. 
 
A lot of my comments are hinting at that direction, but overall I would recommend: 
- Simplifying whenever possible. If complexity is needed, it needs to be explained / documented. This is particularly true of nested multiprocessing layers, which can be hard to build a mental model for.
- Trying to improve codebase uniformity (naming conventions, basic abstractions, comment formats, logging format, etc) to the extent possible.
- Using type hints for arguments and return values whenever possible.
- Moving to `utils` and `const` utilities and values that are shared across commands.
- Making variable and function names more specific when possible. 

---

Finally, here's a tip to avoid having to write certain Peewee queries: Whenever you use a `ForeignKeyField`, objects come with `{table}_set` getters that let you easily retrieve connected elements, regardless of their hierarchy.

So if have 
```python 
import peewee 

class A(peewee.Model): 
     class Meta:
        table_name = "a"
        database = get_db()

    id = peewee.PrimaryKeyField()


class B(peewee.Model): 
     class Meta:
        table_name = "b"
        database = get_db()

    id = peewee.PrimaryKeyField()
    
    a = peewee.ForeignKeyField(model=A, field=A.id)
```

I can do the following: 
```python3
for a in A.select().iterator(): 
  bs = a.b_set # Retrieves a list[B] where B.a.id == a.id
```
"""
