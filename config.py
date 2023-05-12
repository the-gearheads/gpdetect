from pydantic import BaseModel, Field, root_validator, conlist, constr, conint
from devtools import debug
import rtoml
import os

class Nt(BaseModel):
  is_server: bool = False
  port: int = Field(default=5810, lt=65536, ge=0)
  nt3_port: int = Field(default=1735, lt=65536, ge=0)
  address: str = ""
  team_number: int = Field(default=0, lt=10000, ge=0)
  enabled_default_value: bool = False

  @root_validator
  def exclusive_team_num_and_server(cls, values):
    if values["is_server"] and values["team_number"] != 0:
      raise ValueError("is_server and team_number are mutally excluisive")
    return values

  @root_validator
  def nt3_nt4_combined_server_only(cls, values):
    if (not values["is_server"]) and values["nt3_port"] != 1735 and values["port"] != 5810:
      raise ValueError("Can't configure both NT3 and NT4 ports at the same time unless server")
    return values

  @root_validator
  def exclusive_team_num_and_ip(cls, values):
    if values["address"] != "" and values["team_number"] != 0:
      raise ValueError("address and team_number are mutually exclusive")
    return values

class Cam(BaseModel):
  id: int = 0
  resolution: list[int] = [640, 480]
  refresh_rate: int = 30
  fourcc: conlist(constr(to_upper=True, max_length=1), min_items=4, max_items=4) = ['M','J','P','G'] # type: ignore

class Detector(BaseModel):
  model_path = "./conecube.onnx"
  use_ultralytics_v8: bool = False
  conf_threshold: float = 0.5
  iou_threshold: float = 0.5

class Stream(BaseModel):
  enabled: bool = True
  imshow_output: bool = False
  image_scale_factor: float = Field(default=0.25, le=1.0, gt=0)
  jpeg_enc_quality: int = Field(default=50)
  listen_addr: str = "0.0.0.0"
  listen_port: int = Field(default=1189, lt=65536, ge=0)

class Config(BaseModel):
  detector: Detector = Detector()
  nt = Nt()
  camera = Cam()
  stream = Stream()

def load_config(config_path:str = "./config.toml") -> Config:
  # Create default config if it doesn't exist
  if not os.path.exists(config_path):
    with open(config_path, "w") as file:
      rtoml.dump(Config().dict(), file)

  with open(config_path, "r") as file:
    parsed = rtoml.load(file)
    val = Config.parse_obj(parsed)
    return val
