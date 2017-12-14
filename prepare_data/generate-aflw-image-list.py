import ipdb
import os
import numpy as np
import sqlite3
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as plp

N_LANDMARKS = 21

def sqlite_query(conn, select_str, from_str, where_str=None):
    query_str = "SELECT {} FROM {} WHERE {}".format(
        select_str, from_str, where_str)
    return pd.read_sql_query(query_str, conn)


def process_aflw_raw(sqlite_file, image_dir):

    dataset = []

    conn = sqlite3.connect(sqlite_file)

    table_names = sqlite_query(conn, 'name', 'sqlite_master', "type = 'table'")

    select_str = (
        "faces.face_id, "
        "imgs.filepath, "
        "rect.x, rect.y, rect.w, rect.h, "
        "pose.roll, pose.pitch, pose.yaw, "
        "metadata.sex, metadata.occluded, metadata.glasses")
    from_str = (
        "faces, "
        "faceimages imgs, "
        "facerect rect, "
        "facepose pose, "
        "facemetadata metadata")
    where_str = (
        "faces.file_id = imgs.file_id AND "
        "faces.face_id = rect.face_id AND "
        "faces.face_id = pose.face_id AND "
        "faces.face_id = metadata.face_id")
    query_basic = sqlite_query(conn, select_str, from_str, where_str)

    for it, row in enumerate(query_basic.itertuples()):

        # Images that can't be loaded with PIL
        if row.face_id in [41355, 47825, 52945, 60315, 60316, 61583, 62433, 62988, 62989, 63075]:
            continue

        select_str = (
            "faces.face_id, "
            "coords.x, coords.y, coords.feature_id")
        from_str = (
            "faces, "
            "featurecoords coords")
        where_str = (
            "faces.face_id = {} AND "
            "faces.face_id = coords.face_id").format(row.face_id)
        query_landmarks = sqlite_query(conn, select_str, from_str, where_str)

        # left and right in reference to the viewer / image (not the person)
        # left eye == person's right eye (it's mirrored)
        # feature_id = description
        # 1 = left eyebrow - left
        # 2 = left eyebrow - center
        # 3 = left eyebrow - right
        # 4 = right eyebrow - left
        # 5 = right eyebrow - center
        # 6 = right eyebrow - right
        # 7 = left eye - left
        # 8 = left eye - center
        # 9 = left eye - right
        # 10 = right eye - left
        # 11 = right eye - center
        # 12 = right eye - right
        # 13 = left earlobe
        # 14 = nose - left
        # 15 = nose - center
        # 16 = nose - right
        # 17 = right earlobe
        # 18 = mouth - left
        # 19 = mouth - center
        # 20 = mouth - right
        # 21 = chin
        landmarks_x = np.zeros(N_LANDMARKS) * np.nan
        landmarks_y = np.zeros(N_LANDMARKS) * np.nan
        for lm in query_landmarks.itertuples():
            landmarks_x[lm.feature_id - 1] = lm.x
            landmarks_y[lm.feature_id - 1] = lm.y

        filepath = os.path.join(image_dir, row.filepath)

        entry = {
            'image_id': filepath,
            'x_1': row.x,
            'y_1': row.y,
            'width': row.w,
            'height': row.h,
            'lefteye_x': landmarks_x[7],
            'lefteye_y': landmarks_y[7],
            'righteye_x': landmarks_x[10],
            'righteye_y': landmarks_y[10],
            'nose_x': landmarks_x[14],
            'nose_y': landmarks_y[14],
            'leftmouth_x': landmarks_x[17],
            'leftmouth_y': landmarks_y[17],
            'rightmouth_x': landmarks_x[19],
            'rightmouth_y': landmarks_y[19],
        }
        dataset.append(entry)

        if False:
            img = Image.open(entry['image_id']).convert('RGB')
            plt.imshow(np.array(img))
            ax = plt.gca()
            plt.scatter(entry['lefteye_x'], entry['lefteye_y'])
            plt.scatter(entry['righteye_x'], entry['righteye_y'])
            plt.scatter(entry['nose_x'], entry['nose_y'])
            plt.scatter(entry['leftmouth_x'], entry['leftmouth_y'])
            plt.scatter(entry['rightmouth_x'], entry['rightmouth_y'])
            rect = plp.Rectangle(
                (entry['x_1'], entry['y_1']),
                entry['width'], entry['height'],
                fill=False, color='red', linewidth=6)
            ax.add_patch(rect)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    columns = ['image_id', 'x_1', 'y_1', 'width', 'height', 
               'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y',
               'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y',
               'rightmouth_x, rightmouth_y']
    dataset = pd.DataFrame(dataset, columns=columns)
    return dataset

if __name__ == '__main__':
    sqlite_file = 'aflw_image_folder/aflw.sqlite'
    image_dir = 'aflw_image_folder/flickr'

    aflw_df = process_aflw_raw(sqlite_file, image_dir)
    aflw_df.to_csv('aflw_image_list.txt', index=False)
