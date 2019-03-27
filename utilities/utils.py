def adapt_z_position(bbox):
    # bounding boxes from lidar_simulation have position z=0, but bounding boxes in this project use z=h/2
    height = bbox["size"][:, 2]
    position = bbox["position"]
    position[:, 2] = height * .5
    bbox["position"] = position
    return bbox


def on_colab():
    try:
        import google.colab
        return True
    except:
        return False
