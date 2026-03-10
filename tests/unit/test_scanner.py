"""Unit tests for NetworkScanner (all network I/O mocked)."""

import socket
from unittest.mock import patch, MagicMock

import pytest
import networkx as nx

from blackpanther.core.scanner import (
    NetworkScanner,
    ScanResult,
    PortResult,
    SERVICE_MAP,
    COMMON_PORTS,
)
from blackpanther.core.access import AccessPropagation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_connect_factory(open_map):
    """Return a side_effect for socket.connect that uses *open_map*.

    ``open_map`` maps ``(host, port)`` -> True/False.  Missing keys
    raise ``ConnectionRefusedError``.
    """
    def _connect(addr):
        if open_map.get(addr, False):
            return None
        raise ConnectionRefusedError(f"refused {addr}")
    return _connect


# ---------------------------------------------------------------------------
# Host discovery
# ---------------------------------------------------------------------------

class TestDiscoverHosts:

    @patch("blackpanther.core.scanner.socket.socket")
    def test_finds_alive_hosts(self, mock_socket_cls):
        """Hosts that accept a probe connection are discovered."""
        alive_ips = {"10.0.0.1", "10.0.0.5"}

        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        def connect_side_effect(addr):
            host, _port = addr
            if host in alive_ips:
                return None
            raise ConnectionRefusedError

        mock_sock.connect.side_effect = connect_side_effect

        scanner = NetworkScanner(timeout=0.1)
        found = scanner.discover_hosts("10.0.0.0/29", probe_ports=[80])

        assert "10.0.0.1" in found
        assert "10.0.0.5" in found
        assert len(found) == 2

    @patch("blackpanther.core.scanner.socket.socket")
    def test_no_alive_hosts(self, mock_socket_cls):
        """All probes refused -> empty list."""
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect.side_effect = ConnectionRefusedError

        scanner = NetworkScanner(timeout=0.1)
        found = scanner.discover_hosts("10.0.0.0/30", probe_ports=[80])
        assert found == []

    @patch("blackpanther.core.scanner.socket.socket")
    def test_results_populated(self, mock_socket_cls):
        """Discovered hosts appear in scanner.results."""
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect.return_value = None  # all alive

        scanner = NetworkScanner(timeout=0.1)
        scanner.discover_hosts("10.0.0.0/30", probe_ports=[22])

        for ip in scanner.alive_hosts:
            assert scanner.results[ip].alive is True


# ---------------------------------------------------------------------------
# Port scanning
# ---------------------------------------------------------------------------

class TestScanPorts:

    @patch("blackpanther.core.scanner.socket.socket")
    def test_open_ports_detected(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock

        open_ports = {22, 80, 443}
        def connect(addr):
            _, port = addr
            if port in open_ports:
                return None
            raise ConnectionRefusedError

        mock_sock.connect.side_effect = connect

        scanner = NetworkScanner(timeout=0.1)
        result = scanner.scan_ports("10.0.0.1", ports=[22, 80, 443, 3306])

        found = {p.port for p in result.open_ports}
        assert found == {22, 80, 443}

    @patch("blackpanther.core.scanner.socket.socket")
    def test_service_names_assigned(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect.return_value = None

        scanner = NetworkScanner(timeout=0.1)
        result = scanner.scan_ports("10.0.0.1", ports=[22, 3306])

        assert result.services[22] == "ssh"
        assert result.services[3306] == "mysql"

    @patch("blackpanther.core.scanner.socket.socket")
    def test_os_hint_windows(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect.return_value = None

        scanner = NetworkScanner(timeout=0.1)
        result = scanner.scan_ports("10.0.0.1", ports=[135, 445, 3389])
        assert result.os_hint == "windows"

    @patch("blackpanther.core.scanner.socket.socket")
    def test_os_hint_linux(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect.return_value = None

        scanner = NetworkScanner(timeout=0.1)
        result = scanner.scan_ports("10.0.0.1", ports=[22, 80])
        assert result.os_hint == "linux/unix"

    @patch("blackpanther.core.scanner.socket.socket")
    def test_no_open_ports(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect.side_effect = ConnectionRefusedError

        scanner = NetworkScanner(timeout=0.1)
        result = scanner.scan_ports("10.0.0.1", ports=[22, 80])
        assert result.open_ports == []
        assert result.alive is False


# ---------------------------------------------------------------------------
# Service detection (banner grab)
# ---------------------------------------------------------------------------

class TestDetectService:

    @patch("blackpanther.core.scanner.socket.socket")
    def test_banner_ssh(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect.return_value = None
        mock_sock.recv.return_value = b"SSH-2.0-OpenSSH_8.9\r\n"

        scanner = NetworkScanner(timeout=0.1)
        scanner._results["10.0.0.1"] = ScanResult(
            ip="10.0.0.1", alive=True,
            open_ports=[PortResult(port=22, open=True, service="ssh")],
            services={22: "ssh"},
        )

        banner = scanner.detect_service("10.0.0.1", 22)
        assert "SSH" in banner

    @patch("blackpanther.core.scanner.socket.socket")
    def test_banner_updates_service(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect.return_value = None
        mock_sock.recv.return_value = b"220 mail.example.com ESMTP Postfix\r\n"

        scanner = NetworkScanner(timeout=0.1)
        scanner._results["10.0.0.1"] = ScanResult(
            ip="10.0.0.1", alive=True,
            open_ports=[PortResult(port=25, open=True, service="smtp")],
            services={25: "smtp"},
        )

        scanner.detect_service("10.0.0.1", 25)
        assert scanner.results["10.0.0.1"].services[25] == "smtp"

    @patch("blackpanther.core.scanner.socket.socket")
    def test_banner_empty_on_timeout(self, mock_socket_cls):
        mock_sock = MagicMock()
        mock_socket_cls.return_value = mock_sock
        mock_sock.connect.side_effect = socket.timeout

        scanner = NetworkScanner(timeout=0.1)
        banner = scanner.detect_service("10.0.0.1", 22)
        assert banner == ""


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

class TestBuildNetworkGraph:

    def _scanner_with_hosts(self):
        scanner = NetworkScanner(timeout=0.1)
        scanner._results = {
            "192.168.1.1": ScanResult(
                ip="192.168.1.1", alive=True,
                open_ports=[PortResult(22, True, "ssh"),
                            PortResult(80, True, "http")],
                services={22: "ssh", 80: "http"},
            ),
            "192.168.1.2": ScanResult(
                ip="192.168.1.2", alive=True,
                open_ports=[PortResult(22, True, "ssh"),
                            PortResult(3306, True, "mysql")],
                services={22: "ssh", 3306: "mysql"},
            ),
            "10.0.0.1": ScanResult(
                ip="10.0.0.1", alive=True,
                open_ports=[PortResult(80, True, "http")],
                services={80: "http"},
            ),
        }
        return scanner

    def test_graph_has_all_alive_nodes(self):
        scanner = self._scanner_with_hosts()
        G = scanner.build_network_graph()
        assert set(G.nodes()) == {"192.168.1.1", "192.168.1.2", "10.0.0.1"}

    def test_same_subnet_edge(self):
        scanner = self._scanner_with_hosts()
        G = scanner.build_network_graph()
        assert G.has_edge("192.168.1.1", "192.168.1.2")

    def test_cross_subnet_shared_service_edge(self):
        scanner = self._scanner_with_hosts()
        G = scanner.build_network_graph()
        # 192.168.1.1 and 10.0.0.1 both have http
        assert G.has_edge("192.168.1.1", "10.0.0.1")

    def test_edge_attributes(self):
        scanner = self._scanner_with_hosts()
        G = scanner.build_network_graph()
        edge = G.edges["192.168.1.1", "192.168.1.2"]
        assert 0 < edge["weight"] <= 1.0
        assert 0 < edge["vulnerability"] <= 1.0

    def test_empty_scanner_produces_empty_graph(self):
        scanner = NetworkScanner(timeout=0.1)
        G = scanner.build_network_graph()
        assert len(G.nodes()) == 0


# ---------------------------------------------------------------------------
# Integration with AccessPropagation
# ---------------------------------------------------------------------------

class TestIntegrateWithAccess:

    def test_hosts_added_to_model(self):
        scanner = NetworkScanner(timeout=0.1)
        scanner._results = {
            "10.0.0.1": ScanResult(
                ip="10.0.0.1", alive=True,
                open_ports=[PortResult(22, True, "ssh")],
                services={22: "ssh"},
            ),
            "10.0.0.2": ScanResult(
                ip="10.0.0.2", alive=True,
                open_ports=[PortResult(80, True, "http")],
                services={80: "http"},
            ),
        }

        access = AccessPropagation(eta=0.2, mu=0.01)
        scanner.integrate_with_access(access, initial_access_host="10.0.0.1",
                                      initial_access_value=0.3)

        assert "10.0.0.1" in access.hosts
        assert "10.0.0.2" in access.hosts
        assert access.hosts["10.0.0.1"].access == 0.3
        assert access.hosts["10.0.0.2"].access == 0.0

    def test_network_graph_set(self):
        scanner = NetworkScanner(timeout=0.1)
        scanner._results = {
            "10.0.0.1": ScanResult(ip="10.0.0.1", alive=True,
                                   open_ports=[PortResult(22, True, "ssh")],
                                   services={22: "ssh"}),
            "10.0.0.2": ScanResult(ip="10.0.0.2", alive=True,
                                   open_ports=[PortResult(22, True, "ssh")],
                                   services={22: "ssh"}),
        }

        access = AccessPropagation(eta=0.2, mu=0.01)
        G = scanner.integrate_with_access(access)

        assert isinstance(G, nx.Graph)
        assert access._network is not None

    def test_step_after_integration(self):
        """After integration the access model can successfully step."""
        scanner = NetworkScanner(timeout=0.1)
        scanner._results = {
            "10.0.0.1": ScanResult(ip="10.0.0.1", alive=True,
                                   open_ports=[PortResult(22, True, "ssh")],
                                   services={22: "ssh"}),
            "10.0.0.2": ScanResult(ip="10.0.0.2", alive=True,
                                   open_ports=[PortResult(22, True, "ssh")],
                                   services={22: "ssh"}),
        }

        access = AccessPropagation(eta=0.2, mu=0.01)
        scanner.integrate_with_access(access, initial_access_host="10.0.0.1",
                                      initial_access_value=0.5)

        state = access.step(knowledge=0.8, attack_intensity=1.0)
        assert state.global_access >= 0


# ---------------------------------------------------------------------------
# Banner identification helpers
# ---------------------------------------------------------------------------

class TestIdentifyFromBanner:

    @pytest.mark.parametrize("banner,port,expected", [
        ("SSH-2.0-OpenSSH_8.9", 22, "ssh"),
        ("220 ProFTPD Server", 21, "ftp"),
        ("220 mail ESMTP Postfix", 25, "smtp"),
        ("HTTP/1.1 200 OK", 80, "http"),
        ("HTTP/1.1 200 OK", 443, "https"),
        ("5.7.42-0ubuntu0.22.04.1 MySQL", 3306, "mysql"),
        ("redis_version:7.0", 6379, "redis"),
        ("MongoDB shell version v6.0", 27017, "mongodb"),
        ("", 9999, "unknown"),
    ])
    def test_identification(self, banner, port, expected):
        assert NetworkScanner._identify_from_banner(banner, port) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
