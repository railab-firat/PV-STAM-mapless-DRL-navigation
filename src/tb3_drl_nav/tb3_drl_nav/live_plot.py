import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

LOG_PATH = os.path.expanduser("~/tb3_drl_logs/phase3/sac_v2_stacked.csv")

def animate(i):
    if not os.path.exists(LOG_PATH): return
    data = pd.read_csv(LOG_PATH)
    if data.empty: return

    plt.cla()
    plt.plot(data['episode'], data['sr_100'], label='Success Rate (Last 100)')
    plt.xlabel('Episode')
    plt.ylabel('SR %')
    plt.title('SAC v2 Stacked Training Progress')
    plt.legend(loc='upper left')
    plt.grid(True)

def main():
    ani = FuncAnimation(plt.gcf(), animate, interval=1000)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
