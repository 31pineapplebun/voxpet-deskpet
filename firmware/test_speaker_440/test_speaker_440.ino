// ============================================================
// 喇叭 440Hz 测试 (XIAO ESP32S3 + MAX98357A)
// 引脚用实测跑通的 D0/D2/D10, SD 拉高接 VIN
// 烧录后应听到稳定的 "嘟——嘟——嘟——" (440Hz, A4 音)
// ============================================================
#include <ESP_I2S.h>

// MAX98357A 接线 (实测跑通方案):
//   BCLK -> D0 (GPIO1)
//   LRC  -> D2 (GPIO3)
//   DIN  -> D10(GPIO9)
//   VIN  -> 5V,  GND -> GND,  SD -> VIN(拉高)
#define PIN_I2S_BCLK 1     // D0
#define PIN_I2S_LRC  3     // D2
#define PIN_I2S_DIN  9     // D10

const uint32_t SAMPLE_RATE = 24000;
const float    FREQ        = 440.0;   // A4 音

I2SClass spkI2S;

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println("\n=== 喇叭 440Hz 测试 ===");

  spkI2S.setPins(PIN_I2S_BCLK, PIN_I2S_LRC, PIN_I2S_DIN);
  if (!spkI2S.begin(I2S_MODE_STD, SAMPLE_RATE,
                    I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("X 喇叭初始化失败");
    while (1) delay(1000);
  }
  Serial.printf("OK 喇叭就绪 @ %luHz, 应听到 440Hz 嘟嘟声\n", SAMPLE_RATE);
  Serial.println("如果没声音: 检查 SD 是否接 VIN, VIN 是否接 5V, 喇叭线是否夹紧");
}

void loop() {
  // 生成 0.3 秒 440Hz 正弦波, 然后静音 0.3 秒, 循环 -> 嘟——嘟——
  static float phase = 0.0;
  const int N = 256;
  int16_t buf[N];

  // 发声 0.3 秒
  int toneSamples = SAMPLE_RATE * 0.3;
  for (int done = 0; done < toneSamples; done += N) {
    for (int i = 0; i < N; i++) {
      buf[i] = (int16_t)(sin(phase) * 8000);   // 振幅 8000 (约 1/4 最大音量)
      phase += 2.0 * PI * FREQ / SAMPLE_RATE;
      if (phase > 2.0 * PI) phase -= 2.0 * PI;
    }
    spkI2S.write((uint8_t*)buf, sizeof(buf));
  }

  // 静音 0.3 秒
  memset(buf, 0, sizeof(buf));
  int silSamples = SAMPLE_RATE * 0.3;
  for (int done = 0; done < silSamples; done += N) {
    spkI2S.write((uint8_t*)buf, sizeof(buf));
  }
}
