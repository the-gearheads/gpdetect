from aiohttp import web, MultipartWriter

# Taken (and lightly modified) from https://github.com/devenpatel2/aiohttp_mjpeg

class StreamHandler:
  def __init__(self, cam):
    self._cam = cam

  async def __call__(self, request):
    my_boundary = 'image-boundary'
    response = web.StreamResponse(
      status=200,
      reason='OK',
      headers={
        'Content-Type': 'multipart/x-mixed-replace;boundary={}'.format(my_boundary)
      }
    )
    await response.prepare(request)
    while True:
      frame = await self._cam.get_frame()
      with MultipartWriter('image/jpeg', boundary=my_boundary) as mpwriter:
        mpwriter.append(frame, {
          'Content-Type': 'image/jpeg'
        }) # type: ignore
        try:
          await mpwriter.write(response, close_boundary=False)
        except ConnectionResetError :
          print("[MJPG] Client connection closed")
          break
      await response.write(b"\r\n")


class MjpegServer:
  def __init__(self, host='0.0.0.0', port=1189):
    self._port = port
    self._host = host
    self._app = web.Application()
    self._runner: web.AppRunner
    self._site: web.TCPSite
    self._cam_routes = []

  def add_stream(self, route, cam):
    route = f"/{route}"
    self._cam_routes.append(route)
    assert hasattr(cam, 'get_frame'), "arg 'cam' should have a 'get_frame' method"
    self._app.router.add_route("GET", f"{route}", StreamHandler(cam)) # type: ignore

  async def start(self):
    #web.run_app(self._app, host=self._host, port=self._port)
    self._runner = web.AppRunner(self._app)
    await self._runner.setup()
    self._site = web.TCPSite(self._runner, self._host, self._port)
    await self._site.start()

  async def stop(self):
    await self._site.stop()
