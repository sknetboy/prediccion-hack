package com.churninsight.controllers;

import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/stats")
public class StatsController {
    private final RestTemplate restTemplate;

    public StatsController(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Value("${python.service.stats.url:http://127.0.0.1:8001/stats}")
    private String pythonStatsUrl;

    @GetMapping
    public ResponseEntity<Map> stats() {
        Map resp = restTemplate.getForObject(pythonStatsUrl, Map.class);
        return ResponseEntity.ok(resp);
    }
}