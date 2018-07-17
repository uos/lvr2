# Daten des Beispielscans

Ordner: /home/exchange/berufschule_2018

## Ordnerstruktur

### scan_annotated_x.x.ply
- Punktwolke des Scans
- Pro Punkt:
  - x, y, z (float) als Position des Punkts
  - reflectance (float) Reflektierwert _ignorieren_
  - x_coords, y_coords (ushort) Position des Punkts im Panorama Bild

### panoramas_fixed
- panorama_x.x.png: Bild mit normaler Kamera
- panorama_channels_x.x: Bild mit Spektralkamera
  - Informationen der einzelnen Spektralkanäle
  - 150 channels (0 = Infrarot, 149 = Violet)

## Auslesen von Informationen
- Punkte aus .ply Datei auslesen
- Bilder aus panoramas_fixed einlesen
- Zu jedem Punkt p:
  - (p.x_coords, p.y_coords) in Panorama Bildern nachschauen
  - Pixelwert in channel0.png bei (p.x_coords, p.y_coords) => Intensität von Farbkanal 0 an Punkt p
