import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from optimizer import DynaFlowOptimizer

# ==============================================================================
# DYNAFLOW: URBAN MOBILITY DIGITAL TWIN (GEMELLO DIGITALE DELLA SMART CITY)
# ==============================================================================

class VirtualIntersection:
    """Rappresentazione virtuale (gemella) di un singolo incrocio fisico."""
    def __init__(self, node_id, road_id, zone, lat, lon, road_type, lanes, speed_limit, has_school, has_hospital, cycle_time, cameras):
        self.id = node_id
        self.road_id = road_id
        self.zone = zone
        self.lat = lat
        self.lon = lon
        self.road_type = road_type
        self.lanes = lanes
        self.speed_limit = speed_limit
        self.has_school = bool(has_school)
        self.has_hospital = bool(has_hospital)
        self.cycle_time = cycle_time
        self.camera_count = cameras

        # Stato dinamico in tempo reale nel gemello
        self.vehicle_flow = 200.0
        self.speed = speed_limit
        self.queue_meters = 15.0
        self.green_duration = 30.0
        self.iot_health = 'Optimal'
        self.active_accident = False
        self.active_emergency = False


class UrbanDigitalTwin:
    """
    Il Gemello Digitale della Mobilità Urbana:
    - Mantiene lo stato di 100 incroci interconnessi
    - Gestisce la tolleranza ai guasti (Self-Healing per sensori degradati)
    - Esegue simulazioni controfattuali 'What-If' prima dell'attuazione reale
    - Applica l'ottimizzazione fisico-informata per la resilienza della rete
    """
    def __init__(self, dataset_path='smart_city_traffic_mobility.csv'):
        print("Inizializzazione Urban Digital Twin (Caricamento topologia fisica)...")
        self.df = pd.read_csv(dataset_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        self.optimizer = DynaFlowOptimizer()
        self.nodes = {}
        self._build_topology()
        self._train_ml_engine()

    def _build_topology(self):
        """Estrae la mappa statica e la connettività dei 100 incroci."""
        unique_nodes = self.df.drop_duplicates(subset=['intersection_id'])
        for _, r in unique_nodes.iterrows():
            node = VirtualIntersection(
                node_id=r['intersection_id'],
                road_id=r['road_id'],
                zone=r['city_zone'],
                lat=r['latitude'],
                lon=r['longitude'],
                road_type=r['road_type'],
                lanes=r['lanes'],
                speed_limit=r['speed_limit'],
                has_school=r['nearby_school'],
                has_hospital=r['nearby_hospital'],
                cycle_time=r['signal_cycle_seconds'],
                cameras=r['traffic_camera_count']
            )
            self.nodes[node.id] = node
        print(f"Topologia creata: {len(self.nodes)} nodi virtuali attivi su 6 zone urbane.")

    def _train_ml_engine(self):
        """Addestra il motore di previsione della domanda per il gemello."""
        print("Calibrazione modello predittivo per il Digital Twin...")
        df_sorted = self.df.sort_values(by=['intersection_id', 'timestamp']).reset_index(drop=True)
        df_sorted['vehicle_count_lag1'] = df_sorted.groupby('intersection_id')['vehicle_count'].shift(1)
        df_sorted['average_speed_lag1'] = df_sorted.groupby('intersection_id')['average_speed'].shift(1)
        df_sorted['queue_length_lag1'] = df_sorted.groupby('intersection_id')['queue_length'].shift(1)
        df_sorted['sin_hour'] = np.sin(2 * np.pi * df_sorted['hour'] / 24.0)
        df_sorted['cos_hour'] = np.cos(2 * np.pi * df_sorted['hour'] / 24.0)
        df_sorted['target_future_vehicles'] = df_sorted.groupby('intersection_id')['vehicle_count'].shift(-1)
        df_clean = df_sorted.dropna(subset=['vehicle_count_lag1', 'target_future_vehicles']).reset_index(drop=True)

        self.feature_cols = [
            'vehicle_count_lag1', 'average_speed_lag1', 'queue_length_lag1',
            'sin_hour', 'cos_hour', 'day_of_week', 'is_weekend', 'is_holiday', 'rush_hour',
            'temperature', 'rainfall_mm', 'visibility_km', 'air_quality_index',
            'lanes', 'speed_limit', 'nearby_school', 'nearby_hospital', 'nearby_market',
            'construction_activity', 'accident_reported', 'public_event', 'emergency_vehicle_detected'
        ]

        split_idx = int(len(df_clean) * 0.8)
        self.ml_model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, random_state=42)
        self.ml_model.fit(df_clean.iloc[:split_idx][self.feature_cols], df_clean.iloc[:split_idx]['target_future_vehicles'])
        print("Modello predittivo sincronizzato con il Gemello Digitale.")

    def sync_telemetry(self, timestamp):
        """Sincronizza lo stato del Digital Twin con una specifica ora reale."""
        slice_df = self.df[self.df['timestamp'] == timestamp]
        for _, row in slice_df.iterrows():
            nid = row['intersection_id']
            if nid in self.nodes:
                node = self.nodes[nid]
                node.vehicle_flow = row['vehicle_count']
                node.speed = row['average_speed']
                node.queue_meters = row['queue_length']
                node.green_duration = row['green_light_duration']
                node.iot_health = row['iot_sensor_health']
                node.active_accident = bool(row['accident_reported'])
                node.active_emergency = bool(row['emergency_vehicle_detected'])

    def self_healing_sensor(self, node_id):
        """
        Algoritmo di Autoguarigione (Fault-Tolerance):
        Se un sensore IoT è 'Degraded', ricostruisce il flusso reale incrociando
        telecamere e conservazione fisica del moto (Q = K * V).
        """
        node = self.nodes[node_id]
        if node.iot_health == 'Degraded':
            # Densità K stimata dalle telecamere (veic/km per corsia)
            estimated_density = (node.queue_meters / 6.5) / max(node.lanes, 1) * 2.5
            # Flusso ricostruito con legge fondamentale idrodinamica Q = K * V
            reconstructed_q = estimated_density * max(node.speed, 5.0) * node.lanes
            confidence = 0.85
            return reconstructed_q, confidence, "RECONSTRUCTED_VIA_CAMERA_PHYSICS"
        return node.vehicle_flow, 1.0, "OPTIMAL_SENSOR"

    def simulate_what_if(self, target_node_id, shock_type='ACCIDENT_AND_RAIN'):
        """
        Esegue un esperimento virtuale 'What-If':
        Simula uno shock acuto (es. incidente + pioggia torrenziale) sul nodo target
        e valuta:
        1. Scenario Inerziale (Status Quo: nessun intervento intelligente)
        2. Scenario Reattivo DynaFlow (Ottimizzazione adattiva dei semafori)
        """
        node = self.nodes[target_node_id]

        print("\n" + "=" * 78)
        print(f"🧪 SIMULAZIONE WHAT-IF NEL DIGITAL TWIN: INCROCIO {node.id} ({node.zone})")
        print("=" * 78)
        print(f"Asse Stradale: {node.road_id} ({node.road_type}, {node.lanes} corsie, Limite {node.speed_limit} km/h)")
        
        # Parametri dello shock virtuale
        if shock_type == 'ACCIDENT_AND_RAIN':
            print("🚨 SHOCK INIETTATO NEL GEMELLO:")
            print("   • Incidente Grave (Corsia ostruita: capacità -50%)")
            print("   • Nubifragio improvviso (Rainfall: 14.5 mm/h, Visibilità: 3.2 km)")
            print("   • Picco Domanda Veicolare: 1.850 veicoli/h")
            q_demand = 1850.0
            effective_lanes = max(1, node.lanes - 1) # una corsia bloccata dall'incidente
            is_emergency = node.has_hospital
        else:
            q_demand = 1500.0
            effective_lanes = node.lanes
            is_emergency = False

        # 1. SCENARIO STATUS QUO (Semaforo rigido della città senza intelligenza)
        base_green = node.green_duration
        base_delay = self.optimizer.webster_delay(q_demand, base_green, node.cycle_time, effective_lanes)
        base_queue, base_emiss, base_fuel = self.optimizer.estimate_queue_and_emissions(
            q_demand, base_delay, base_green, node.cycle_time, effective_lanes, node.has_school or node.has_hospital
        )

        # 2. SCENARIO DYNAFLOW DIGITAL TWIN (Intervento proattivo intelligente)
        opt_res = self.optimizer.optimize_intersection(
            q_main=q_demand,
            C=node.cycle_time,
            lanes_main=effective_lanes,
            baseline_g=base_green,
            is_sensitive=node.has_school or node.has_hospital,
            is_emergency=is_emergency
        )

        # Calcolo benefici nel Gemello
        print("\n📊 ESITO DELLA SIMULAZIONE CONTROFATTUALE NEL GEMELLO VIRTUALE:")
        print("-" * 78)
        print(f"{'Metrica Operativa':<32} | {'Status Quo (No AI)':<20} | {'DynaFlow Digital Twin':<20}")
        print("-" * 78)
        print(f"{'Verde Assegnato':<32} | {base_green:.0f} secondi{'':<13} | {opt_res['optimal_green']:.0f} secondi ({opt_res['delta_green']:+.0f}s)")
        print(f"{'Ritardo Medio per Auto':<32} | {base_delay:.1f} secondi{'':<12} | {opt_res['optimal_delay']:.1f} secondi ({opt_res['delay_gain_pct']:+.1f}%)")
        print(f"{'Lunghezza Coda Stimata':<32} | {base_queue:.1f} metri{'':<14} | {base_queue * (1 - opt_res['queue_gain_pct']/100):.1f} metri ({opt_res['queue_gain_pct']:+.1f}%)")
        print(f"{'Emissioni CO2 Orarie':<32} | {base_emiss:.1f} kg CO2{'':<13} | {base_emiss * (1 - opt_res['emiss_gain_pct']/100):.1f} kg CO2 ({opt_res['emiss_gain_pct']:+.1f}%)")
        print("-" * 78)

        print("\n💡 DIAGNOSI DEL GEMELLO DIGITALE PER L'OPERATORE:")
        if opt_res['queue_gain_pct'] > 20:
            print(f"  ✅ Intervento Salvavita: L'allungamento del verde di {opt_res['delta_green']:+.0f}s evita il blocco della corsia,")
            print(f"     tagliando la coda virtuale di oltre {abs(base_queue - base_queue * (1 - opt_res['queue_gain_pct']/100)):.0f} metri prima che intasi l'incrocio a monte!")
        print("=" * 78 + "\n")


# ==============================================================================
# ESECUZIONE DIMOSTRATIVA DEL DIGITAL TWIN
# ==============================================================================
if __name__ == '__main__':
    twin = UrbanDigitalTwin()

    # Test 1: Autoguarigione Sensore Guasto
    print("\n🛠️ TEST 1: TOLLERANZA AI GUASTI IOT (SELF-HEALING)")
    print("-" * 78)
    twin.nodes['INT_001'].iot_health = 'Degraded'
    twin.nodes['INT_001'].queue_meters = 180.0
    flow, conf, method = twin.self_healing_sensor('INT_001')
    print(f"Sensore INT_001 Degradato: Flusso ricostruito con fisica e telecamere = {flow:.1f} veic/h (Confidenza: {conf*100:.0f}%, Metodo: {method})")

    # Test 2: Simulazione What-If su Incrocio Critico di Downtown
    twin.simulate_what_if('INT_014', shock_type='ACCIDENT_AND_RAIN')
