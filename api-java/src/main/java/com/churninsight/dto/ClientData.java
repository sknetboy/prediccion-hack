package com.churninsight.dto;

import jakarta.validation.constraints.NotNull;

public class ClientData {
    @NotNull
    private Integer tiempo_contrato_meses;
    @NotNull
    private Integer retrasos_pago;
    @NotNull
    private Double uso_mensual;
    @NotNull
    private String plan;
    @NotNull
    private Integer nps;
    @NotNull
    private Integer quejas;
    @NotNull
    private String canal_contacto;
    @NotNull
    private Integer interacciones_soporte;
    @NotNull
    private String tipo_pago;
    @NotNull
    private String region;
    @NotNull
    private String tipo_cliente;

    public Integer getTiempo_contrato_meses() { return tiempo_contrato_meses; }
    public void setTiempo_contrato_meses(Integer v) { this.tiempo_contrato_meses = v; }
    public Integer getRetrasos_pago() { return retrasos_pago; }
    public void setRetrasos_pago(Integer v) { this.retrasos_pago = v; }
    public Double getUso_mensual() { return uso_mensual; }
    public void setUso_mensual(Double v) { this.uso_mensual = v; }
    public String getPlan() { return plan; }
    public void setPlan(String v) { this.plan = v; }
    public Integer getNps() { return nps; }
    public void setNps(Integer v) { this.nps = v; }
    public Integer getQuejas() { return quejas; }
    public void setQuejas(Integer v) { this.quejas = v; }
    public String getCanal_contacto() { return canal_contacto; }
    public void setCanal_contacto(String v) { this.canal_contacto = v; }
    public Integer getInteracciones_soporte() { return interacciones_soporte; }
    public void setInteracciones_soporte(Integer v) { this.interacciones_soporte = v; }
    public String getTipo_pago() { return tipo_pago; }
    public void setTipo_pago(String v) { this.tipo_pago = v; }
    public String getRegion() { return region; }
    public void setRegion(String v) { this.region = v; }
    public String getTipo_cliente() { return tipo_cliente; }
    public void setTipo_cliente(String v) { this.tipo_cliente = v; }
}