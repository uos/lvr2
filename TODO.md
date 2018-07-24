# TODO

- Dateien Laden
  - hdf5 Support
	- speicher sparenderes Verfahren überlegen? _(später, optional)_
  - Tatsächliche Repräsentation _(mit anderen Gruppen absprechen, folgt später)_

- Daten verwenden
  - Einfärbung aktuallisieren _(nicht in jedem Frame)_
    - evlt. erst bei los lassen des Cursors aktualisieren und nicht bei jedem Step

- Benutzerinterface
  - Bedeutung kenntlich machen _(Tooltips, Beschreibung, etc)_
  - Visuell anschaulich gestalten _(sichtbares Spektrum unterm Slider?)_
  - Channel für Farbe deaktivierbar machen? _(z.B. g und b konstant auf 0 und rot auf channel x)_

## Erweiterung I: Gradientendarstellung
- zweiter Tab beim SpectralDiaglog
  - für bestimmten Channel ein Farbgradienten auswählen _(siehe vorhandenen code aus lvr2)_

## Erweiterung II: Punkte auswählen
- Punkt auswählen mithilfe von Interactor->GetPicker()
  - Komplettes Spektrum eines Punktes als Wellenlänge -> Intensität Graph separat darstellen
