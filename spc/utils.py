import numpy as np


def make_video(rollouts, m=4):
    videos = list()
    max_t = 0

    for rollout in rollouts:
        video = list()

        for data in rollout:
            video.append(data.s.transpose(2, 0, 1))

        max_t = max(max_t, len(rollout))
        videos.append(np.uint8(video))

    videos.sort(key=lambda x: x.shape[0], reverse=True)

    _, c, h, w = videos[0].shape
    full = np.zeros((max_t, c, h * m, w * m), dtype=np.uint8)

    for i in range(m):
        for j in range(m):
            if i * m + j >= len(videos):
                continue

            n = videos[i * m + j].shape[0]
            full[:n, :, i * h: i * h + h, j * w: j * w + w] = videos[i * m + j]
            full[n:, :, i * h: i * h + h, j * w: j * w + w] = videos[i * m + j][-1][None]

    return full
