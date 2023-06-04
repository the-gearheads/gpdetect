from ntcore import NetworkTableInstance
from devtools import debug
import cv2
import typing
import asyncio
import config
import mjpg_server

cfg: config.Config = config.load_config()

from deepsparse.pipeline import Pipeline
from deepsparse.yolo.utils import YOLOOutput
from deepsparse.yolact.utils import annotate_image as annotate_image_segmentation
from deepsparse.yolo.utils import annotate_image as annotate_image_detection

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

  model = Pipeline.create(
    task="yolov8",
    subtask="detection",
    model_path=cfg.detector.model_path,
    class_names=["cone", "cube"],
    engine_type="deepsparse",
    image_size=(640, 640),
  )

  detPub = gpDet.getDoubleArrayTopic("Detections").publish()
  enabledSub = gpDet.getBooleanTopic("Enabled").subscribe(cfg.nt.enabled_default_value)
  enabledAck = gpDet.getBooleanTopic("EnabledACK").publish()
  m = cv2.TickMeter()
  while cap.isOpened():
    await asyncio.sleep(0) # Give time for the webserver code to run
    ret, frame = cap.read()

    enabledAck.set(enabledSub.get())
    if not enabledSub.get():
      continue

    if not ret:
      print(f"cap.read returned {ret} :(")
      break

    m.start()

    img_transposed = frame[:, :, ::-1].transpose(2, 0, 1)
    result = typing.cast(YOLOOutput, model(images=[img_transposed]))
    
    boxes = result.boxes
    scores = result.scores
    class_ids = result.labels

    drawn_frame=annotate_image_detection(
      image=frame,
      prediction=result,
      score_threshold=cfg.detector.conf_threshold,
      images_per_sec=m.getFPS(),
    )

    m.stop()

    if cfg.stream.enabled:
      mjpg_handler.update_frame(drawn_frame)

    if cfg.stream.imshow_output:
      cv2.imshow("A", drawn_frame)
      cv2.waitKey(1)

    outArray = []
    for i in range(len(boxes[0])):
     outArray.append(0 if class_ids[0][i] == "cone" else 1)
     outArray.append(scores[0][i])
     for j in range(len(boxes[0][i])//2):
       outArray.append(boxes[0][i][2*j]/xres)
       outArray.append(boxes[0][i][2*j+1]/yres)

    detPub.set(outArray)

if __name__ == "__main__":
    asyncio.run(main())
