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

    videos.sort(key=lambda x: x.shape[0], reverse=True)

    _, c, h, w = videos[0].shape
    full = np.zeros((max_t, c, h * 8, w * 8), dtype=np.uint8)

    for i in range(8):
        for j in range(8):
            if i * 8 + j >= len(videos):
                continue

            n = videos[i * 8 + j].shape[0]
            full[:n, :, i * h: i * h + h, j * w: j * w + w] = videos[i * 8 + j]
            full[n:, :, i * h: i * h + h, j * w: j * w + w] = videos[i * 8 + j][-1][None]

    return full
