import cv2
import os
import sys
import argparse
import pandas as pd
from matplotlib import pyplot as plt

#######################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--base", type=str,
                    default=os.path.abspath(os.path.join(__file__, "../../")))
args = parser.parse_args()
#######################################################################################################################

df = pd.read_csv(os.path.join(args.base, "dfl-bundesliga-data-shootout", "train.csv"))


def check_dir(path: str) -> str:
    """
    check and create dir if not exist
    :param path: path like string
    :return: path
    """
    if os.path.exists(path) and os.path.isdir(path):
        pass
    else:
        check_dir(os.path.abspath(os.path.join(path, "../")))
        os.mkdir(path)
    return path


classDict = {"none": 0,
             "challenge": 1,
             "play": 2,
             "throwin": 3
             }

# temp = []
# for i in range(len(df)):
#     if pd.isna(df.loc[i,'event_attributes']):
#         continue
#     k = eval(df.loc[i,'event_attributes'])
#     if type(k) == list:
#         for item in k:
#             if item not in temp:
#                 temp.append(item)
event_attributes_list = ['ball_action_forced',
                         'opponent_dispossessed',
                         'pass',
                         'openplay',
                         'cross',
                         'possession_retained',
                         'freekick',
                         'fouled',
                         'opponent_rounded',
                         'corner',
                         'challenge_during_ball_transfer']
for attribute in event_attributes_list:
    classDict[attribute] = len(classDict)

print(classDict)

check_dir(os.path.join(args.base, "charades"))
framesPath = check_dir(os.path.join(args.base, "charades", "frames"))
frameConfigPath = check_dir(os.path.join(args.base, "charades", "frame_lists"))
with open(os.path.join(frameConfigPath, "train.txt"), "w", encoding='utf-8') as file:
    pass
with open(os.path.join(frameConfigPath, "label_id.txt"), "w", encoding='utf-8') as file:
    file.write("\n".join(["{}\t{}".format(key, val) for val, key in enumerate(classDict)]))
    file.flush()

cur_video_id = ""
cur_frame_id = 0
cur_video_ind = 0
res = False
camera = cv2.VideoCapture(
    os.path.join(args.base, "dfl-bundesliga-data-shootout", "train", "{}.mp4".format(df.loc[0, "video_id"])))
for i in range(len(df)):
    # 只处理中间值
    if df.loc[i, 'event'] in ['start', 'end']:
        continue
    # 当视频id切换时，处理上一视频的剩余帧数
    if df.loc[i, "video_id"] != cur_video_id:
        cur_frame_id += 1
        cur_video_id = 0
        with open(os.path.join(frameConfigPath, "train.txt"), "a", encoding='utf-8') as file:
            # file.write("{}\t{}\n".format("{}_{}".format(cur_video_id, cur_frame_id),
            #                              df.loc[i, 'event_attributes'] + "," + ",".join(eval(
            #                                  df.loc[i, 'event_attributes'])) if pd.isna(
            #                                  df.loc[i, 'event_attributes']) else df.loc[i, 'event_attributes']))
            file.write("{}\t{}\n".format("{}_{}".format(cur_video_id, cur_frame_id), "none"))
        while res:
            cv2.imwrite(os.path.join(framesPath, "{}_{}".format(cur_video_id, cur_frame_id),
                                     "{}_{}-{}.jpg".format(cur_video_id, cur_frame_id, cur_video_ind)),
                        image[:, :, ::-1])
            res, image = camera.read()
            cur_video_ind += 1
        # 刷新更新参数
        cur_video_id = df.loc[i, "video_id"]
        cur_frame_id = 0
        cur_video_ind = 0
        check_dir(os.path.join(framesPath, "{}_{}".format(cur_video_id, cur_frame_id)))
        camera = cv2.VideoCapture(os.path.join(args.base, "dfl-bundesliga-data-shootout", "train", "{}.mp4".format(cur_video_id)))
        print("{}\t :turn to this new video".format(cur_video_id))

    res, image = camera.read()
    # 帧率
    base_fps = int(round(camera.get(cv2.CAP_PROP_FPS)))
    # 当前帧数
    cur_frame = camera.get(cv2.CAP_PROP_POS_FRAMES)

    cur_frame_id += 1
    cur_video_ind = 0
    if cur_frame / base_fps < df.loc[i, 'time']:
        with open(os.path.join(frameConfigPath, "train.txt"), "a", encoding='utf-8') as file:
            file.write("{}\t{}\n".format("{}_{}".format(cur_video_id, cur_frame_id), "none"))
    while res and cur_frame / base_fps < df.loc[i-1, 'time']:
        cv2.imwrite(os.path.join(framesPath, "{}_{}".format(cur_video_id, cur_frame_id), "{}_{}-{}.jpg".format(cur_video_id, cur_frame_id, cur_video_ind)), image[:, :, ::-1])
        res, image = camera.read()
        cur_video_ind += 1
        # 帧率
        base_fps = int(round(camera.get(cv2.CAP_PROP_FPS)))
        # 当前帧数
        cur_frame = camera.get(cv2.CAP_PROP_POS_FRAMES)

    cur_frame_id += 1
    cur_video_ind = 0
    print("\t{}\t : {}".format("{}_{}".format(cur_video_id, cur_frame_id), df.loc[i, 'event'] + "," + ",".join(eval(df.loc[i, 'event_attributes'])) if not pd.isna(df.loc[i, 'event_attributes']) else df.loc[i, 'event']))
    with open(os.path.join(frameConfigPath, "train.txt"), "a", encoding='utf-8') as file:
        file.write("{}\t{}\n".format("{}_{}".format(cur_video_id, cur_frame_id),
                                     df.loc[i, 'event'] + "," + ",".join(eval(
                                         df.loc[i, 'event_attributes'])) if not pd.isna(
                                         df.loc[i, 'event_attributes']) else df.loc[i, 'event']))
    while res and cur_frame / base_fps < df.loc[i+1, 'time']:
        cv2.imwrite(os.path.join(framesPath, "{}_{}".format(cur_video_id, cur_frame_id), "{}_{}-{}.jpg".format(cur_video_id, cur_frame_id, cur_video_ind)), image[:, :, ::-1])
        res, image = camera.read()
        cur_video_ind += 1
        # 帧率
        base_fps = int(round(camera.get(cv2.CAP_PROP_FPS)))
        # 当前帧数
        cur_frame = camera.get(cv2.CAP_PROP_POS_FRAMES)


