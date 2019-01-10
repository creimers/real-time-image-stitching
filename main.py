import timeit

import cv2
import numpy as np


class Matcher:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, i1, i2):
        image_set_1 = self.get_SURF_features(i1)
        image_set_2 = self.get_SURF_features(i2)
        matches = self.flann.knnMatch(image_set_2["des"], image_set_1["des"], k=2)
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            points_current = image_set_2["kp"]
            points_previous = image_set_1["kp"]

            matched_points_current = np.float32(
                [points_current[i].pt for (__, i) in good]
            )
            matched_points_prev = np.float32(
                [points_previous[i].pt for (i, __) in good]
            )

            H, _ = cv2.findHomography(
                matched_points_current, matched_points_prev, cv2.RANSAC, 4
            )
            return H
        return None

    def get_SURF_features(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return {"kp": kp, "des": des}


class Stitcher:
    def __init__(
        self,
        number_of_images,
        crop_x_min=None,
        crop_x_max=None,
        crop_y_min=None,
        crop_y_max=None,
    ):

        self.matcher_obj = Matcher()
        self.homography_cache = {}

        self.count = number_of_images

        self.crop_x_min = crop_x_min
        self.crop_x_max = crop_x_max
        self.crop_y_min = crop_y_min
        self.crop_y_max = crop_y_max

    def stitch(self, images=[]):
        """
        stitches the images into a panorama
        """
        # TODO this transformation might not be necessary
        self.images = [cv2.cvtColor(i, cv2.COLOR_RGB2RGBA) for i in images]

        self.prepare_lists()

        # left stitching
        start = timeit.default_timer()
        self.left_shift()
        self.right_shift()
        stop = timeit.default_timer()
        duration = stop - start
        print("stitching took %.2f seconds." % duration)

        if self.crop_x_min and self.crop_x_max and self.crop_y_min and self.crop_y_max:
            return self.result[
                self.crop_y_min : self.crop_y_max, self.crop_x_min : self.crop_x_max
            ]
        else:
            return self.result

    def prepare_lists(self):

        # reset lists
        self.left_list = []
        self.right_list = []

        self.center_index = int(self.count / 2)

        self.result = self.images[self.center_index]

        for i in range(self.count):
            if i <= self.center_index:
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])

    def get_homography(self, image_1, image_1_key, image_2, image_2_key, direction):
        # TODO: use image indexes from the input array
        """
        Calculate the homography matrix between two images.
        Return from cache if possible.

        Args:
            image_1 (np.array) - first image
            image_1_key (str) - identifier for cache
            image_2 (np.array) - second image
            image_2_key (str) - identifier for cache
            direction (str) - "left" or "right"
        Returns:
            homography (np.array) - Homograpy Matrix
        """

        cache_key = "_".join([image_1_key, image_2_key, direction])
        homography = self.homography_cache.get(cache_key, None)
        if homography is None:
            # TODO: is the homography the same regardless of order??
            homography = self.matcher_obj.match(image_1, image_2)
            # put in cache
            self.homography_cache[cache_key] = homography
        return homography

    def left_shift(self):
        """
        stitch images center to left
        """
        # start off with center image
        a = self.left_list[0]

        for i, image in enumerate(self.left_list[1:]):
            H = self.get_homography(a, str(i), image, str(i + 1), "left")

            # inverse homography
            XH = np.linalg.inv(H)

            ds = np.dot(XH, np.array([a.shape[1], a.shape[0], 1]))
            ds = ds / ds[-1]

            f1 = np.dot(XH, np.array([0, 0, 1]))
            f1 = f1 / f1[-1]

            XH[0][-1] += abs(f1[0])
            XH[1][-1] += abs(f1[1])

            ds = np.dot(XH, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))

            # dimension of warped image
            dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)

            tmp = cv2.warpPerspective(a, XH, dsize, borderMode=cv2.BORDER_TRANSPARENT)

            # punch the image in there
            tmp[
                offsety : image.shape[0] + offsety, offsetx : image.shape[1] + offsetx
            ] = image

            a = tmp

        self.result = tmp

    def right_shift(self):
        """
        stitch images center to right
        """
        for i, imageRight in enumerate(self.right_list):
            imageLeft = self.result

            H = self.get_homography(imageLeft, str(i), imageRight, str(i + 1), "right")

            # args: original_image, matrix, output shape (width, height)
            result = cv2.warpPerspective(
                imageRight,
                H,
                (imageLeft.shape[1] + imageRight.shape[1], imageLeft.shape[0]),
                borderMode=cv2.BORDER_TRANSPARENT,
            )

            mask = np.zeros((result.shape[0], result.shape[1], 4), dtype="uint8")
            mask[0 : imageLeft.shape[0], 0 : imageLeft.shape[1]] = imageLeft
            mask_rgb = mask[:, :, :3]  # Grab the BRG planes
            result = self.blend_transparent(mask_rgb, result)

            # TODO maybe this step can be put in to blend_transparent?
            self.result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)

    def blend_transparent(self, background, foreground):
        """
        code obtained from here: 
        https://stackoverflow.com/questions/36921496/how-to-join-png-with-alpha-transparency-in-a-frame-in-realtime/37198079#37198079
        """
        # Split out the transparency mask from the colour info
        overlay_img = foreground[:, :, :3]  # Grab the BRG planes
        overlay_mask = foreground[:, :, 3:]  # And the alpha plane

        # Again calculate the inverse mask
        background_mask = 255 - overlay_mask

        # Turn the masks into three channel, so we can use them as weights
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

        # Create a masked out face image, and masked out overlay
        # We convert the images to floating point in range 0.0 - 1.0
        background_part = (background * (1 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

        # And finally just add them together, and rescale it back to an 8bit integer image
        return np.uint8(
            cv2.addWeighted(background_part, 255.0, overlay_part, 255.0, 0.0)
        )


if __name__ == "__main__":
    FRAME_WIDTH = 768
    FRAME_HEIGHT = 432

    shanghai_files = [
        "images/shanghai-01.png",
        "images/shanghai-02.png",
        "images/shanghai-03.png",
        "images/shanghai-04.png",
        "images/shanghai-05.png",
    ]

    shanghai = [
        cv2.resize(cv2.imread(f), (FRAME_WIDTH, FRAME_HEIGHT)) for f in shanghai_files
    ]

    crop_x_min = 30
    crop_x_max = 1764
    crop_y_min = 37
    crop_y_max = 471

    s = Stitcher(
        len(shanghai_files),
        crop_x_min=crop_x_min,
        crop_x_max=crop_x_max,
        crop_y_min=crop_y_min,
        crop_y_max=crop_y_max,
    )

    panorama = s.stitch(shanghai)

    cv2.imwrite("panorama.png", panorama)
