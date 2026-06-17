#include <SD.h>
//const int chipSelect = 4;

#define BAUDRATE 57600

// neuro data variables
byte errorRate = 200;
byte heartrate = 0;
short raw;


// system variables

boolean newRawData = false;
boolean bigPacket = false;

//////////////////////////
// Microprocessor Setup //
//////////////////////////
void setup() {

  Serial.begin(BAUDRATE);           // USB
}



/////////////
//MAIN LOOP//
/////////////
void loop() {
  ReadData();

//    if (newRawData) {
//      newRawData = false;
//      
//      SD.begin(chipSelect);
//      File dataFile = SD.open("RawLog.txt", FILE_WRITE);
//      
//         if (dataFile) {
//            dataFile.print(" RAW: ");
//            dataFile.println(raw, DEC);
//       
//            dataFile.close();
//          }
//    }
     
//  if(bigPacket) {
//    bigPacket = false;        
//      SD.begin(chipSelect);
//      File dataFile = SD.open("Log.txt", FILE_WRITE);
//      
//         if (dataFile) {
//            dataFile.print("PoorQuality: ");
//            dataFile.print(errorRate);
//            dataFile.print(" HeartRate: ");
//            dataFile.println(heartrate);
//       
//            dataFile.close();
//          }
//  }
      
      
      
      
      if(newRawData) {
         
        newRawData = false;
        Serial.print(errorRate);
        Serial.print(",");
        Serial.print(heartrate);
        Serial.print(",");
        Serial.println(raw, DEC);
       }
}//stop loop


////////////////////////////////
// Read data from Serial UART //
////////////////////////////////
byte ReadOneByte() {
  int ByteRead;

  while(!Serial.available());
  ByteRead = Serial.read();

  return ByteRead;
}

/////////////////////////////////////////
// Reading data from MindSet bluetooth //
////////////////////////////////////////
void ReadData() {
  static unsigned char payloadData[256]; 
  byte generatedChecksum;
  byte checksum; 
  byte vLength;
  int payloadLength;
  int k;

  
  // Look for sync bytes
  if(ReadOneByte() == 170) {
    if(ReadOneByte() == 170) {

      do { payloadLength = ReadOneByte(); }
      while (payloadLength == 170);
      
      if(payloadLength > 169) {    //Payload length can not be greater than 170
         return;
      }
      
      generatedChecksum = 0;        
      for(int i = 0; i < payloadLength; i++) {  
        payloadData[i] = ReadOneByte();            //Read payload into memory
        generatedChecksum += payloadData[i];
      }   

      checksum = ReadOneByte();                      //Read checksum byte from stream      
      generatedChecksum = 255 - generatedChecksum;   //Take one's compliment of generated checksum
      
      if(checksum != generatedChecksum) {  
        // checksum error  
      } else {  

        for(int i = 0; i < payloadLength; i++) {    // Parse the payload
          switch (payloadData[i]) {
          case 2:
            bigPacket = true;            
            i++;            
            errorRate = payloadData[i];         
            break;
          case 3:
            i++;
            heartrate = payloadData[i]; 
            i = i + 14;            
            break;
          case 0x80: // raw data
            newRawData = true;
            i++;
            vLength = payloadData[i]; 
            raw = 0;
            for (int j=0; j<vLength; j++) {
              raw = raw | ( payloadData[i+vLength-j]<<(8*j) ); // bit-shift little-endian
            }
            i += vLength;
            break;
          default:
            break;
          } // switch
        } // for loop
      } // checksum success
    } // sync 2
  } // sync 1
} // ReadData

