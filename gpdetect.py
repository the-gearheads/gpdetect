from ntcore import NetworkTableInstance
from devtools import debug
import cv2
import yolov7
import config

cfg: config.Config = config.load_config()

def setup_nt() -> NetworkTableInstance:
  ntInst = NetworkTableInstance.getDefault()
  nt = cfg.nt
  if nt.is_server:
    listen_addr = nt.address
    if listen_addr == "":
      listen_addr = "0.0.0.0"
    ntInst.startServer(listen_address=listen_addr, port3=nt.nt3_port, port4=nt.port)
  else:
    use_nt3 = nt.port == 0 or (nt.port == 5810 and nt.nt3_port != 1735)
    use_team_number = nt.team_number != 0
    connect_addr = nt.address
    if not use_team_number and connect_addr == "":
      connect_addr = "127.0.0.1"
    port = 0
    if use_nt3:
      port = nt.nt3_port
    else:
      port = nt.port
    if use_team_number:
      ntInst.setServerTeam(nt.team_number, port)
    else:
      ntInst.setServer(nt.address, port)
    if use_nt3:
      ntInst.startClient3("GPDetect")
    else:
      ntInst.startClient4("GPDetect")
  return ntInst


def main():
  gpDet = setup_nt().getTable("GPDetect")
  cap: cv2.VideoCapture = cv2.VideoCapture(cfg.camera.id)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(cfg.camera.fourcc[0], cfg.camera.fourcc[1], cfg.camera.fourcc[2], cfg.camera.fourcc[3]))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.resolution[0])
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.resolution[1])
  cap.set(cv2.CAP_PROP_FPS, cfg.camera.refresh_rate)
  xres = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  yres = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print(f"Camera running at {xres}x{yres}@{int(cap.get(cv2.CAP_PROP_FPS))}fps")
  
  yolo_model = yolov7.YOLOv7(cfg.detector.model_path, cfg.detector.conf_threshold, cfg.detector.iou_threshold)
  detPub = gpDet.getDoubleArrayTopic("Detections").publish()
  while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
      print(f"cap.read returned {ret} :(")
      break

    boxes, scores, class_ids = yolo_model(frame)
    drawn_frame = yolo_model.draw_detections(frame)
    cv2.imshow("aaa", drawn_frame)

    outArray = []
    for i in range(len(boxes)):
      outArray.append(class_ids[i])
      outArray.append(scores[i])
      for j in range(len(boxes[i])//2):
        outArray.append(boxes[i][2*j]/xres)
        outArray.append(boxes[i][2*j+1]/yres)

    detPub.set(outArray)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if __name__ == "__main__":
    main()
