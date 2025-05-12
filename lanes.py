import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def averaged_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    output_lines = []
    # only average if we actually found left lines
    if left_fit:
        left_avg = np.average(left_fit, axis=0)
        output_lines.append(make_coordinates(image, left_avg))
    # only average if we actually found right lines
    if right_fit:
        right_avg = np.average(right_fit, axis=0)
        output_lines.append(make_coordinates(image, right_avg))

    return np.array(output_lines)


def canny(image):
    # make the image grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # smooth the image and reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # canny edge detection
    return cv2.Canny(blur, 50, 150)


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None and len(lines):
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


# creating the vertices for the region of interest
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    # take the bitwise & of each of the pixels in both the arrays
    # to show only the region of interest
    return cv2.bitwise_and(image, mask)


# read the image
# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(
#     cropped_image, 2, np.pi/180, 100, np.array([]),
#     minLineLength=40, maxLineGap=5
# )
# averaged_lines = averaged_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('result', combined_image)
# cv2.waitKey(0)


cap = cv2.VideoCapture('test2.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(
        cropped_image, 2, np.pi/180, 100, np.array([]),
        minLineLength=40, maxLineGap=5
    )

    if lines is not None:
        averaged_lines = averaged_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow('result', combined_image)
    else:
        # no lines detected → just show the raw frame
        cv2.imshow('result', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()