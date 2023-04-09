import matplotlib.pyplot as plt
import numpy as np

with open('scores.txt', 'r') as f:
    scores = f.read().splitlines()
    scores = list(map(int, scores))

    mean = []
    max_list = []
    for i,j in enumerate(scores):
        if i % 1000 == 0:
            mean.append(np.average(scores[i:i+1000]))
            max_list.append(max(scores[i:i+1000]))

    episode = range(len(mean))
    plt.ylabel('episode scores')
    plt.xlabel('training episodes')
    plt.yticks(np.arange(min(mean), max(mean)+1, 5000))
    plt.title('mean scores')
    plt.plot(episode, mean)
    plt.show()

    plt.ylabel('episode scores')
    plt.xlabel('training episodes')
    plt.yticks(np.arange(min(max_list), max(max_list)+1, 10000))
    plt.title('max scores')
    plt.plot(episode, max_list)
    plt.show()



