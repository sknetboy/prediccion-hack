package com.churninsight.dto;

import java.util.List;

public class PredictResponse {
    private String prevision;
    private double probabilidad;
    private List<String> explicacion;

    public String getPrevision() { return prevision; }
    public void setPrevision(String v) { this.prevision = v; }
    public double getProbabilidad() { return probabilidad; }
    public void setProbabilidad(double v) { this.probabilidad = v; }
    public List<String> getExplicacion() { return explicacion; }
    public void setExplicacion(List<String> e) { this.explicacion = e; }
}