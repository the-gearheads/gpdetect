from ntcore import NetworkTableInstance
from devtools import debug
import cv2
import asyncio
import config
import mjpg_server

cfg: config.Config = config.load_config()

# Kinda takes a while to import so do it conditionally
if cfg.detector.use_ultralytics_v8:
  from ultralytics import YOLO
else:
  import yolov7

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


class CamHandler:
  stored_frame: cv2.Mat
  async def get_frame(self):
    await asyncio.sleep(0) # This somehow makes it work faster
    img = self.stored_frame
    if cfg.stream.image_scale_factor != 1:
      img = cv2.resize(self.stored_frame, (0, 0), fx=cfg.stream.image_scale_factor, fy=cfg.stream.image_scale_factor, interpolation=cv2.INTER_LINEAR) # type: ignore
    frame = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, cfg.stream.jpeg_enc_quality])[1]
    return frame.tobytes()

  def update_frame(self, new_frame):
    self.stored_frame = new_frame

async def main():
  gpDet = setup_nt().getTable("GPDetect")
  cap: cv2.VideoCapture = cv2.VideoCapture(cfg.camera.id)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(cfg.camera.fourcc[0], cfg.camera.fourcc[1], cfg.camera.fourcc[2], cfg.camera.fourcc[3]))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.resolution[0])
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.resolution[1])
  cap.set(cv2.CAP_PROP_FPS, cfg.camera.refresh_rate)
  cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
  xres = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  yres = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print(f"Camera running at {xres}x{yres}@{int(cap.get(cv2.CAP_PROP_FPS))}fps")

  mjpg_handler = CamHandler()
  if cfg.stream.enabled:
    serv = mjpg_server.MjpegServer()
    mjpg_handler.update_frame(cap.read()[1])
    serv.add_stream("", mjpg_handler)
    await serv.start()

  # This makes my IDE angry but it works
  if cfg.detector.use_ultralytics_v8:
    yolo_model = YOLO(cfg.detector.model_path, task="detect")
  else:
    yolo_model = yolov7.YOLOv7(cfg.detector.model_path, cfg.detector.conf_threshold, cfg.detector.iou_threshold)

  detPub = gpDet.getDoubleArrayTopic("Detections").publish()
  enabledSub = gpDet.getBooleanTopic("Enabled").subscribe(cfg.nt.enabled_default_value)
  enabledAck = gpDet.getBooleanTopic("EnabledACK").publish()
  while cap.isOpened():
    await asyncio.sleep(0) # Give time for the webserver code to run
    cv2.waitKey(1)
    ret, frame = cap.read()

    enabledAck.set(enabledSub.get())
    if not enabledSub.get():
      continue

    if not ret:
      print(f"cap.read returned {ret} :(")
      break

    boxes, scores, class_ids = [None, None, None]
    if cfg.detector.use_ultralytics_v8:
      results = yolo_model.predict(frame, conf=cfg.detector.conf_threshold, iou=cfg.detector.iou_threshold) # type: ignore
      boxes = results[0].boxes.xyxy
      scores = results[0].boxes.conf
      class_ids = results[0].boxes.cls
      drawn_frame = results[0].plot()
    else:
      boxes, scores, class_ids = yolo_model(frame)
      drawn_frame = yolo_model.draw_detections(frame)

    if cfg.stream.enabled:
      mjpg_handler.update_frame(drawn_frame)

    if cfg.stream.imshow_output:
      cv2.imshow("A", drawn_frame)

    outArray = []
    for i in range(len(boxes)):
      outArray.append(class_ids[i])
      outArray.append(scores[i])
      for j in range(len(boxes[i])//2):
        outArray.append(boxes[i][2*j]/xres)
        outArray.append(boxes[i][2*j+1]/yres)

    detPub.set(outArray)

if __name__ == "__main__":
    asyncio.run(main())
