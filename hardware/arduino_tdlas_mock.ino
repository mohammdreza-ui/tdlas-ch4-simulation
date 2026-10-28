/*
 * Arduino Uno Beer-Lambert Liquid Absorption Mock with LED Modulation
 *
 * Demonstrates:
 * - Square-wave LED driver modulation (~500 Hz on Pin 9)
 * - Synchronous ADC sampling of transmitted light intensity on A0
 * - Serial streaming of high/low phases to PC for lock-in demodulation
 *
 * Safety Warning: Never look directly into laser or high-power LED beams.
 */

const int LED_PIN = 9;
const int SENSOR_PIN = A0;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  // 500 Hz modulation (1000 microseconds HIGH, 1000 microseconds LOW)
  digitalWrite(LED_PIN, HIGH);
  delayMicroseconds(1000);
  int val_high = analogRead(SENSOR_PIN);
  
  digitalWrite(LED_PIN, LOW);
  delayMicroseconds(1000);
  int val_low = analogRead(SENSOR_PIN);
  
  // Stream ADC readings (high_phase, low_phase)
  Serial.print(val_high);
  Serial.print(",");
  Serial.println(val_low);
}
