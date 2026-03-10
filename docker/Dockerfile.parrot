# BlackPanther V2 - ParrotOS Security Edition
# For real hacking with all tools pre-installed

FROM parrotsec/security:latest

LABEL maintainer="ReignAgent Research"
LABEL description="BlackPanther V2 - AI Penetration Testing Agent"
LABEL version="2.0.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /blackpanther

# Install Python 3.11 and essential tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    python3.11-distutils \
    # Network tools
    nmap \
    masscan \
    netcat-traditional \
    netcat-openbsd \
    hydra \
    john \
    sqlmap \
    nikto \
    gobuster \
    dirb \
    wfuzz \
    # Exploitation tools
    metasploit-framework \
    exploitdb \
    searchsploit \
    # Web tools
    burpsuite \
    zaproxy \
    # Utilities
    curl \
    wget \
    git \
    vim \
    tmux \
    zsh \
    htop \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Create virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /blackpanther/data/exploits /blackpanther/data/vector_store /blackpanther/logs

# Update exploit database
RUN searchsploit -u

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "=========================================="\n\
echo "BlackPanther V2 - AI Penetration Testing"\n\
echo "Version: 2.0.0"\n\
echo "Architecture: $(uname -m)"\n\
echo "=========================================="\n\
\n\
# Activate virtual environment\n\
source /venv/bin/activate\n\
\n\
# Load environment variables\n\
if [ -f "/blackpanther/.env" ]; then\n\
    set -a\n\
    source /blackpanther/.env\n\
    set +a\n\
    echo "✅ Environment loaded"\n\
fi\n\
\n\
# Show available tools\n\
echo "🔍 Security tools ready:"\n\
command -v nmap >/dev/null 2>&1 && echo "  ✅ nmap" || echo "  ❌ nmap"\n\
command -v msfconsole >/dev/null 2>&1 && echo "  ✅ metasploit" || echo "  ⚠️  metasploit"\n\
command -v searchsploit >/dev/null 2>&1 && echo "  ✅ exploitdb" || echo "  ⚠️  exploitdb"\n\
command -v hydra >/dev/null 2>&1 && echo "  ✅ hydra" || echo "  ⚠️  hydra"\n\
\n\
echo "=========================================="\n\
echo "Container ready for BlackPanther V2"\n\
echo "=========================================="\n\
\n\
exec "$@"\n' > /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/zsh"]