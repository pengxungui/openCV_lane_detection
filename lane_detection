import cv2
import numpy as np

##图片灰化、高斯过滤以及canny
image = cv2.imread("E:/Python/lane/data/lane_2.jpg")
image = cv2.resize(image,(720,540),interpolation=cv2.INTER_CUBIC)
cv2.imshow("laneWindow",image)
gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
cv2.imshow("grayWindow",gray_image)
Gaus_image = cv2.GaussianBlur(gray_image,(5,5),0,0)
cv2.imshow("GaussianWindow",Gaus_image)
edge_image = cv2.Canny(Gaus_image,50,200)
cv2.imshow("EdgeWindow",edge_image)

##设置ROI
def roi(image,region):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,region,255)
    roi_image = cv2.bitwise_and(image,mask)
    return roi_image

roi_vtc = np.array([[(0,edge_image.shape[0]-20),(340,320),
                     (350,320),(edge_image.shape[1]-100,edge_image.shape[0]-20)]])
roi_image = roi(edge_image,roi_vtc)
cv2.imshow("roiWindow",roi_image)

##找出直线斜率
def draw_lanes(img, lines, color=[0, 0, 255], thickness=8):
  left_lines, right_lines = [], []
  for line in lines:
    for x1, y1, x2, y2 in line:
      k = (y2 - y1) / (x2 - x1)
      if k < 0:
        left_lines.append(line)
      else:
        right_lines.append(line)
  
  if (len(left_lines) <= 0 or len(right_lines) <= 0):
    return img
  
  clean_lines(left_lines, 0.1)
  clean_lines(right_lines, 0.1)
  left_points = [(x1, y1) for line in left_lines for x1,y1,x2,y2 in line]
  left_points = left_points + [(x2, y2) for line in left_lines for x1,y1,x2,y2 in line]
  right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]
  right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]
  
  left_vtx = calc_lane_vertices(left_points, img.shape[0])
  right_vtx = calc_lane_vertices(right_points, img.shape[0])
  
  cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
  cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)
  
def clean_lines(lines, threshold):
  slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
  while len(lines) > 0:
    mean = np.mean(slope)
    diff = [abs(s - mean) for s in slope]
    idx = np.argmax(diff)
    if diff[idx] > threshold:
      slope.pop(idx)
      lines.pop(idx)
    else:
      break
  
  
def calc_lane_vertices(point_list, ymax):
  x = [p[0] for p in point_list]
  y = [p[1] for p in point_list]
  fit = np.polyfit(y, x, 1)
  fit_fn = np.poly1d(fit)
  ymin = min(y)
  xmin = int(fit_fn(ymin))
  xmax = int(fit_fn(ymax))
  
  return [(xmin, ymin), (xmax, ymax)]

##霍夫变换
def hough_trans(image,rho,theta,threshold,min_line_length,max_line_gap):
    lines = cv2.HoughLinesP(image,rho,theta,threshold,
                            minLineLength = min_line_length,
                            maxLineGap = max_line_gap)
    line_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
##    for line in lines:
##        for x1,y1,x2,y2 in line:
##            cv2.line(line_image,(x1,y1),(x2,y2),[0,0,255],2)
    draw_lanes(line_image,lines)
    return line_image

line_image = hough_trans(roi_image,1,np.pi/180,10,40,20)
cv2.imshow("lineWindow",line_image)
lane_image = cv2.addWeighted(image,0.71,line_image,1,0)
cv2.imshow("lane_Window",lane_image)
cv2.imwrite("E:/Python/lane/data/lane_1_result.jpg",lane_image)
cv2.imwrite("E:/Python/lane/data/lane_1_gray.jpg",gray_image)
cv2.imwrite("E:/Python/lane/data/lane_1_edge.jpg",edge_image)
cv2.imwrite("E:/Python/lane/data/lane_1_roi.jpg",roi_image)
cv2.imwrite("E:/Python/lane/data/lane_1_line.jpg",line_image)
cv2.waitKey()
cv2.destroyAllWindows()
