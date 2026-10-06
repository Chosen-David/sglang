from .base import Indexer
from .tia_indexer import TIAIndexer
from .quest_indexer import QuestIndexer
from .twi_indexer import TwilightIndexer
from .tli_indexer import TLIIndexer

indexer_type_dict = {
    'quest': QuestIndexer,
    'tia': TIAIndexer,
    'twi': TwilightIndexer,
    'tli': TLIIndexer,
}
