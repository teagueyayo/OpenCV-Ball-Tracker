import numpy as np
import cv2.cv2 as cv2
import sys
import matplotlib

# bgr value that represents the ball
bgr_color = 50,20,133
# HSV Error margin of the colours
color_threshold = 15
frames = 348
xStart = 100
pathLength = 1169
pixelsPerFrame = pathLength/348

hsv_color = cv2.cvtColor( np.uint8([[bgr_color]] ), cv2.COLOR_BGR2HSV)[0][0]
HSV_lower = np.array([hsv_color[0] - color_threshold, hsv_color[1] - color_threshold, hsv_color[2] - color_threshold])
HSV_upper = np.array([hsv_color[0] + color_threshold, hsv_color[1] + color_threshold, hsv_color[2] + color_threshold])

minRadius = 25
xList = [] 
xPath = []
xPredicted = []
variance = []
sensorVariance = 6255.98
def detect_ball(frame):
    x, y, radius = -1, -1, -1
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, HSV_lower, HSV_upper)
    mask = cv2.erode(mask, None, iterations=0)
    mask = cv2.dilate(mask, None, iterations=12)
    im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = (-1, -1)

    # only proceed if at least one contour was found
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        xList.append(x) # x coordinates
        xPath.append((x-xStart)/pathLength) # path traveled 
        M = cv2.moments(mask)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # check that the radius is larger than some threshold
        if radius > minRadius:
            #outline ball
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            #show ball center
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

    return center[0], center[1], radius


if __name__ == "__main__":

    frameCounter = 0
    detectedFrame = 0
    p_tt = 0
    x_tt = xStart
    filepath = sys.argv[1]
    cap = cv2.VideoCapture(filepath)
    cv2.namedWindow('frame', flags=(cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO + cv2.WINDOW_GUI_EXPANDED))
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frameCounter += 1
        frameDisplacemenet = frameCounter - detectedFrame
        #frame_resized = cv2.resize(frame, (500, 500))

        sensorX, sensorY, radius = detect_ball(frame)
        # simple implementation of a Kalman filter
        cv2.imshow('frame', frame)
        if radius > minRadius:
            x_t1t = x_tt - (frameDisplacemenet * pixelsPerFrame)
            p_t1t = p_tt + 1
            z_t1 = x_t1t 
            r_t1 = sensorX - z_t1
            s_t1 = p_t1t + sensorVariance # difference sensor variance for each video
            k_t1 = p_t1t *(1.0/s_t1)
            x_tt = x_t1t + k_t1 * r_t1
            p_tt = p_t1t - (p_t1t *1.0 * (1.0/s_t1) * 1.0 * p_t1t)

            xPredicted.append(x_tt) #predicted x coordinate
            variance.append(p_tt) # variance

        # Display the resulting frame
        
        
        if radius != -1:
            detectedFrame = frameCounter #in case of frame skipping
        if sensorX == xStart + pathLength: #close video once path traveled
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    # When everything done, release the capture
    np.savetxt("xList.csv", xList, delimiter = ",")
    np.savetxt("xPath.csv", xPath, delimiter = ",")
    np.savetxt("xPredicted.csv", xPredicted,  delimiter=",")
    np.savetxt("variance.csv", variance, delimiter=",")
    cap.release()
    cv2.destroyAllWindows()


