import numpy as np


def make_video(rollouts):
    videos = list()
    max_t = 0

    for rollout in rollouts:
        video = list()

        for data in rollout:
            video.append(data.s.transpose(2, 0, 1))

        max_t = max(max_t, len(rollout))
        videos.append(np.uint8(video))

    _, c, h, w = videos[0].shape

    full = np.zeros((max_t, c, h * 4, w * 4), dtype=np.uint8)

    for i in range(4):
        for j in range(4):
            n = videos[i * 4 + j].shape[0]
            full[:n, :, i * h: i * h + h, j * w: j * w + w] = videos[i * 4 + j]

    return full
