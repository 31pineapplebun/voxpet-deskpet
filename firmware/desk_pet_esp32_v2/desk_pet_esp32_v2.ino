// ============================================================
// 桌宠 v2 - ESP32S3 端 (XIAO ESP32S3 SENSE + MAX98357A)
// 按住按键说话 -> PCM 上传 -> 等待 -> 收到合成音频 -> 播放
//
// 协议: REC_START/REC_END 上传, TTS_START/TTS_END 下发,
//       中间用二进制帧传 16kHz/16bit/单声道 PCM。
//
// 需要的库 (Arduino 库管理器搜索安装):
//   - WebSockets by Markus Sattler (arduinoWebSockets)
//   - ESP_I2S (ESP32 核心自带)
// ============================================================
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ESP_I2S.h>

// ============ 改成你自己的 ============
const char*    WIFI_SSID = "YOUR_WIFI_SSID";        // WiFi 名 (ESP32 仅支持 2.4GHz)
const char*    WIFI_PASS = "YOUR_WIFI_PASSWORD";    // WiFi 密码
const char*    WS_HOST   = "192.168.1.100";         // 跑 desk_pet_server.py 的那台 PC 的局域网 IP
const uint16_t WS_PORT   = 8765;

// ============ 引脚 (实测跑通方案!) ============
// 按键用宏 D1 (=GPIO2), 音频 BCLK 用数字 1 (=D0=GPIO1)
// 注意: 宏 D1 和 数字1 是不同引脚! D1=GPIO2, 数字1=GPIO1
//   之前踩过坑: 按键和BCLK都写数字会冲突, 必须按键用宏D1
#define PIN_BUTTON   D1      // 扩展板 User 按键 (按下=LOW), D1=GPIO2

// MAX98357A -> XIAO (实测跑通: D0/D2/D10 + SD拉高VIN)
#define PIN_I2S_BCLK 1       // D0 = GPIO1 (数字1, 不是宏D1!)
#define PIN_I2S_LRC  3       // D2 = GPIO3
#define PIN_I2S_DIN  9       // D10 = GPIO9

// 板载 PDM 麦克 (固定, 别改)
#define PIN_PDM_CLK  42
#define PIN_PDM_DATA 41

// ============ 音频参数 ============
const uint32_t MIC_RATE    = 16000;   // 麦克 16k, Whisper 友好
const uint32_t SPK_RATE    = 16000;   // 喇叭也 16k (PC 端把 SoVITS 的 32k 重采样到 16k 再发来)
const size_t   CHUNK_BYTES = 1024;

// ============ 全局对象 ============
I2SClass micI2S;
I2SClass spkI2S;
WebSocketsClient ws;

enum State { IDLE, RECORDING, RECEIVING };
volatile State state = IDLE;

// ============ WebSocket 事件 ============
void onWsEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      Serial.printf("OK WS 已连接: %s\n", payload);
      state = IDLE;
      break;
    case WStype_DISCONNECTED:
      Serial.println("X WS 已断开");
      state = IDLE;
      break;
    case WStype_TEXT:
      if (length == 9 && memcmp(payload, "TTS_START", 9) == 0) {
        Serial.println(">> 接收音频...");
        state = RECEIVING;
      } else if (length == 7 && memcmp(payload, "TTS_END", 7) == 0) {
        Serial.println(">> 播放完成\n");
        state = IDLE;
      }
      break;
    case WStype_BIN:
      if (state == RECEIVING) {
        spkI2S.write(payload, length);   // 收到 PCM 直接播放
      }
      break;
    default:
      break;
  }
}

// ============ Setup ============
void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println("\n=== 桌宠 v2 启动 ===");

  pinMode(PIN_BUTTON, INPUT_PULLUP);

  // 麦克风 (PDM RX) @ 16kHz
  micI2S.setPinsPdmRx(PIN_PDM_CLK, PIN_PDM_DATA);
  if (!micI2S.begin(I2S_MODE_PDM_RX, MIC_RATE,
                    I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("X 麦克风初始化失败");
    while (1) delay(1000);
  }
  Serial.printf("OK 麦克风就绪 @ %luHz\n", MIC_RATE);

  // 喇叭 (Std I2S TX)
  spkI2S.setPins(PIN_I2S_BCLK, PIN_I2S_LRC, PIN_I2S_DIN);
  if (!spkI2S.begin(I2S_MODE_STD, SPK_RATE,
                    I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("X 喇叭初始化失败");
    while (1) delay(1000);
  }
  Serial.printf("OK 喇叭就绪 @ %luHz\n", SPK_RATE);

  // WiFi
  Serial.printf("连接 WiFi: %s ", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  WiFi.setSleep(false);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(300); Serial.print(".");
    if (++tries > 60) { Serial.println("\nX WiFi 连不上, 检查热点名/密码"); tries = 0; }
  }
  Serial.printf("\nOK IP: %s\n", WiFi.localIP().toString().c_str());

  // WebSocket
  ws.begin(WS_HOST, WS_PORT, "/");
  ws.onEvent(onWsEvent);
  ws.setReconnectInterval(3000);
  ws.enableHeartbeat(15000, 3000, 2);

  Serial.println("\n按住 BUTTON 键说话, 松开发送...\n");
}

// ============ Loop ============
void loop() {
  ws.loop();

  static bool lastButton = HIGH;
  bool button = digitalRead(PIN_BUTTON);

  // 按下 -> 开始录音
  if (button == LOW && lastButton == HIGH && state == IDLE) {
    Serial.println(">> 录音中...");
    state = RECORDING;
    ws.sendTXT("REC_START");
  }

  // 按住 -> 持续上传 PCM
  if (state == RECORDING && button == LOW) {
    static uint8_t buf[CHUNK_BYTES];
    size_t n = micI2S.readBytes((char*)buf, CHUNK_BYTES);
    if (n > 0) ws.sendBIN(buf, n);
  }

  // 松开 -> 结束, 等回复
  if (button == HIGH && lastButton == LOW && state == RECORDING) {
    Serial.println(">> 发送结束, 等待回复...");
    ws.sendTXT("REC_END");
    state = IDLE;
  }

  lastButton = button;
}
