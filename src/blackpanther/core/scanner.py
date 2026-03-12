"""Network Scanner Module

Performs host discovery, port scanning, and service detection,
then feeds discovered topology into the AccessPropagation model.

Uses only Python builtins (socket, ipaddress) -- no external dependencies
beyond networkx which is already in the project.
"""

import socket
import ipaddress
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .access import AccessPropagation

logger = logging.getLogger(__name__)

COMMON_PORTS = [21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143,
                443, 445, 993, 995, 1723, 3000, 3306, 3389, 5432,
                5900, 6379, 8000, 8080, 8443, 8888, 9000, 27017]

SERVICE_MAP: Dict[int, str] = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
    80: "http", 110: "pop3", 111: "rpcbind", 135: "msrpc",
    139: "netbios-ssn", 143: "imap", 443: "https", 445: "smb",
    993: "imaps", 995: "pop3s", 1723: "pptp", 3000: "http",
    3306: "mysql", 3389: "rdp", 5432: "postgresql", 5900: "vnc",
    6379: "redis", 8000: "http", 8080: "http-proxy",
    8443: "https-alt", 8888: "http", 9000: "http", 27017: "mongodb",
}


@dataclass
class PortResult:
    """Result of scanning a single port.

    Attributes:
        port: Port number
        open: Whether the port accepted a connection
        service: Identified service name
        banner: Raw banner bytes grabbed from the service
        response_time: Round-trip connect time in seconds
    """
    port: int
    open: bool
    service: str = ""
    banner: str = ""
    response_time: float = 0.0


@dataclass
class ScanResult:
    """Aggregated scan result for one host.

    Attributes:
        ip: Host IP address string
        alive: Whether the host responded to any probe
        open_ports: List of PortResult for open ports
        services: Mapping of port -> service name
        os_hint: Rough OS guess from open-port fingerprint
        scan_time: Total wall-clock scan duration in seconds
    """
    ip: str
    alive: bool = False
    open_ports: List[PortResult] = field(default_factory=list)
    services: Dict[int, str] = field(default_factory=dict)
    os_hint: str = "unknown"
    scan_time: float = 0.0


class NetworkScanner:
    """TCP connect scanner with host discovery and service detection.

    Example::

        scanner = NetworkScanner(timeout=1.0, max_threads=50)
        scanner.discover_hosts("192.168.1.0/24")
        for host in scanner.results:
            scanner.scan_ports(host, ports=[22, 80, 443])
        graph = scanner.build_network_graph()

        access = AccessPropagation(eta=0.2, mu=0.01)
        scanner.integrate_with_access(access)
    """

    def __init__(
        self,
        timeout: float = 1.0,
        max_threads: int = 50,
    ):
        """
        Args:
            timeout: Socket connect timeout in seconds per probe.
            max_threads: Maximum concurrent threads for scanning.
        """
        self.timeout = timeout
        self.max_threads = max_threads
        self._results: Dict[str, ScanResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_hosts(
        self,
        subnet: str,
        probe_ports: Optional[List[int]] = None,
    ) -> List[str]:
        """Sweep a CIDR subnet and return IPs that respond.

        For each address in the range a TCP connect is attempted on
        ``probe_ports`` (defaults to 80, 443, 22).  Any successful
        connect marks the host as alive.

        Args:
            subnet: CIDR notation, e.g. ``"192.168.1.0/24"``.
            probe_ports: Ports used for the liveness probe.

        Returns:
            List of IP strings that responded.
        """
        if probe_ports is None:
            probe_ports = [80, 443, 22]

        network = ipaddress.ip_network(subnet, strict=False)
        hosts = [str(addr) for addr in network.hosts()]

        alive: List[str] = []

        with ThreadPoolExecutor(max_workers=self.max_threads) as pool:
            future_to_ip = {
                pool.submit(self._probe_host, ip, probe_ports): ip
                for ip in hosts
            }
            for future in as_completed(future_to_ip):
                ip = future_to_ip[future]
                try:
                    is_alive = future.result()
                except Exception:
                    is_alive = False

                if is_alive:
                    alive.append(ip)
                    if ip not in self._results:
                        self._results[ip] = ScanResult(ip=ip, alive=True)
                    else:
                        self._results[ip].alive = True

        logger.info("Discovered %d alive hosts in %s", len(alive), subnet)
        return sorted(alive, key=ipaddress.ip_address)

    def scan_ports(
        self,
        host: str,
        ports: Optional[List[int]] = None,
    ) -> ScanResult:
        """TCP-connect scan a host on the given ports.

        Args:
            host: IP address or hostname.
            ports: Port list (defaults to ``COMMON_PORTS``).

        Returns:
            ``ScanResult`` with open-port details populated.
        """
        if ports is None:
            ports = list(COMMON_PORTS)

        t0 = time.monotonic()

        if host not in self._results:
            self._results[host] = ScanResult(ip=host)

        result = self._results[host]
        open_ports: List[PortResult] = []

        with ThreadPoolExecutor(max_workers=self.max_threads) as pool:
            future_to_port = {
                pool.submit(self._tcp_connect, host, port): port
                for port in ports
            }
            for future in as_completed(future_to_port):
                port = future_to_port[future]
                try:
                    is_open, rtt = future.result()
                except Exception:
                    is_open, rtt = False, 0.0

                if is_open:
                    service_name = SERVICE_MAP.get(port, "unknown")
                    pr = PortResult(
                        port=port,
                        open=True,
                        service=service_name,
                        response_time=rtt,
                    )
                    open_ports.append(pr)

        open_ports.sort(key=lambda p: p.port)
        result.open_ports = open_ports
        result.alive = bool(open_ports)
        result.services = {p.port: p.service for p in open_ports}
        result.os_hint = self._guess_os(result)
        result.scan_time = time.monotonic() - t0

        return result

    def detect_service(self, host: str, port: int) -> str:
        """Grab the service banner from an open port.

        Args:
            host: Target IP / hostname.
            port: Open port number.

        Returns:
            Banner string (may be empty if the service sends nothing).
        """
        banner = self._grab_banner(host, port)

        if host in self._results:
            for pr in self._results[host].open_ports:
                if pr.port == port:
                    pr.banner = banner
                    if banner:
                        pr.service = self._identify_from_banner(banner, port)
                        self._results[host].services[port] = pr.service

        return banner

    def build_network_graph(self) -> nx.Graph:
        """Build a NetworkX graph from scan results.

        Nodes are alive hosts.  Two hosts share an edge when they have
        at least one common open service **or** are on the same /24
        subnet.  Edge weight reflects the number of shared services
        (more shared services -> stronger link).  A ``vulnerability``
        attribute is set proportional to high-risk services found.

        Returns:
            ``nx.Graph`` ready for ``AccessPropagation.set_network()``.
        """
        G = nx.Graph()

        alive = {ip: res for ip, res in self._results.items() if res.alive}
        for ip, res in alive.items():
            G.add_node(ip, services=list(res.services.values()),
                       os_hint=res.os_hint)

        ips = list(alive.keys())
        high_risk_services = {"ftp", "telnet", "smb", "rdp", "vnc",
                              "redis", "mongodb", "mysql", "postgresql"}

        for i in range(len(ips)):
            for j in range(i + 1, len(ips)):
                ip_a, ip_b = ips[i], ips[j]
                svc_a = set(alive[ip_a].services.values())
                svc_b = set(alive[ip_b].services.values())
                shared = svc_a & svc_b

                same_subnet = (
                    ipaddress.ip_network(f"{ip_a}/24", strict=False)
                    == ipaddress.ip_network(f"{ip_b}/24", strict=False)
                )

                if shared or same_subnet:
                    weight = max(0.1, min(1.0, len(shared) * 0.2 + (0.3 if same_subnet else 0.0)))
                    risky = (svc_a | svc_b) & high_risk_services
                    vuln = max(0.1, min(1.0, len(risky) * 0.15))
                    G.add_edge(ip_a, ip_b, weight=weight, vulnerability=vuln)

        return G

    def integrate_with_access(
        self,
        access: AccessPropagation,
        initial_access_host: Optional[str] = None,
        initial_access_value: float = 0.2,
    ) -> nx.Graph:
        """Populate an ``AccessPropagation`` model from scan results.

        Adds every alive host and wires them with the topology graph.

        Args:
            access: Target model instance.
            initial_access_host: IP of the foothold host (gets nonzero access).
            initial_access_value: Starting access on the foothold host.

        Returns:
            The ``nx.Graph`` that was applied.
        """
        G = self.build_network_graph()

        for ip, res in self._results.items():
            if not res.alive:
                continue
            init = initial_access_value if ip == initial_access_host else 0.0
            services = list(res.services.values())
            access.add_host(ip, initial_access=init, services=services)

        access.set_network(G)
        return G

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def results(self) -> Dict[str, ScanResult]:
        """Copy of current scan results keyed by IP."""
        return dict(self._results)

    @property
    def alive_hosts(self) -> List[str]:
        """IPs that have been marked alive."""
        return sorted(
            [ip for ip, r in self._results.items() if r.alive],
            key=ipaddress.ip_address,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _probe_host(self, ip: str, ports: List[int]) -> bool:
        """Return True if any port on *ip* accepts a TCP connection."""
        for port in ports:
            is_open, _ = self._tcp_connect(ip, port)
            if is_open:
                return True
        return False

    def _tcp_connect(self, host: str, port: int) -> Tuple[bool, float]:
        """Attempt a TCP connect and return (open, rtt_seconds)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        t0 = time.monotonic()
        try:
            sock.connect((host, port))
            rtt = time.monotonic() - t0
            return True, rtt
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False, 0.0
        finally:
            sock.close()

    def _grab_banner(self, host: str, port: int) -> str:
        """Connect and read up to 1024 bytes within timeout."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect((host, port))
            sock.sendall(b"\r\n")
            data = sock.recv(1024)
            return data.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
        finally:
            sock.close()

    @staticmethod
    def _identify_from_banner(banner: str, port: int) -> str:
        """Heuristic service identification from banner text."""
        bl = banner.lower()
        if "ssh" in bl:
            return "ssh"
        if "ftp" in bl:
            return "ftp"
        if "smtp" in bl or "postfix" in bl or "exim" in bl:
            return "smtp"
        if "http" in bl or "html" in bl or "nginx" in bl or "apache" in bl:
            return "https" if port == 443 else "http"
        if "mysql" in bl or "mariadb" in bl:
            return "mysql"
        if "postgresql" in bl:
            return "postgresql"
        if "redis" in bl:
            return "redis"
        if "mongo" in bl:
            return "mongodb"
        return SERVICE_MAP.get(port, "unknown")

    @staticmethod
    def _guess_os(result: ScanResult) -> str:
        """Very rough OS guess based on open-port profile."""
        ports = {p.port for p in result.open_ports}
        if ports & {135, 139, 445, 3389}:
            return "windows"
        if ports & {22}:
            return "linux/unix"
        return "unknown"
