import sys
import os

# Fix paths for imports
sys.path.append('/content/WIP/GameFormer')
sys.path.append('/content/WIP/GameFormer/interaction_prediction')

# Fix argument parsing for local rank
sys.argv = [a.replace('--local-rank', '--local_rank') for a in sys.argv]

# Run GameFormer's training script
exec(open('/content/WIP/GameFormer/interaction_prediction/train.py').read())