'use strict';
const $ = (id) => document.getElementById(id);
let selected = null;
let stationChosen = false;
let current = null;
let eventSignature = '';
let stationSignature = '';
let peerSignature = '';
let toastTimer;
let updatingControls = false;
const modes = {oak: 'LIVE OAK', replay: 'RECORDED FOOTAGE', none: 'RELAY STATION'};

function toast(text) {
  $('toast').textContent = text;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {$('toast').hidden = true;}, 4000);
}
async function post(path, body) {
  const response = await fetch(path, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
    signal: AbortSignal.timeout(5000),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}
function item(tag, text, className) {
  const element = document.createElement(tag);
  if (text != null) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function chooseStation(id) {
  stationChosen = true;
  selected = id;
  $('station').value = id;
  if (current) render(current);
}
function render(data) {
  current = data;
  $('connection').textContent = 'Dashboard connected';
  $('connection').className = 'connection';
  const nodes = data.nodes;
  if (!selected || !nodes.some(n => n.id === selected) ||
      (!stationChosen && !nodes.find(n => n.id === selected)?.fresh)) {
    selected = (nodes.find(n => n.fresh) || nodes.find(n => n.id === data.local_id) || nodes[0])?.id;
  }
  const signature = nodes.map(n => n.id + n.name + n.online).join('|');
  if (signature !== stationSignature) {
    $('station').replaceChildren();
    for (const node of nodes) {
      const option = item('option', node.name + (node.online ? '' : ' · offline'));
      option.value = node.id;
      $('station').append(option);
    }
    stationSignature = signature;
  }
  $('station').value = selected || '';
  const node = nodes.find(n => n.id === selected);
  const local = nodes.find(n => n.id === data.local_id);
  $('online').textContent = nodes.filter(n => n.online).length;
  $('in-view').textContent = node?.fresh ? node.detections.length : '—';
  $('sightings').textContent = data.events.length;
  $('latency').textContent = node?.fresh && node.latency_ms != null ? Math.round(node.latency_ms) + ' ms' : '—';
  $('local-name').textContent = local?.name || '';
  $('camera-status').textContent = local?.detail || '';
  $('footer-state').textContent = nodes.filter(n => n.online).length + ' station(s) online · Local network';
  if (!updatingControls && document.activeElement !== $('confidence')) {
    $('source').value = data.source;
    $('confidence').value = Math.round(data.confidence * 100);
    $('confidence-value').textContent = Math.round(data.confidence * 100) + '%';
  }
  $('address').value = data.urls[0] || 'Network address unavailable';
  $('copy').disabled = !data.urls.length;
  const isReplay = node?.source === 'replay';
  if (node) {
    $('origin').textContent = node.name;
    $('frame-age').textContent = node.frame_age == null ? 'No frame yet' : 'Updated ' + Math.round(node.frame_age) + 's ago';
    $('source-note').textContent = isReplay ? 'Replay of saved OAK footage' + (node.captured_at ? ' · ' + new Date(node.captured_at * 1000).toLocaleDateString() : '') : node.source === 'none' ? 'Receiving observations from connected stations' : 'OAK camera · RT-DETR-L';
    const hasFrame = node.captured_at != null;
    $('feed').hidden = !hasFrame;
    $('placeholder').hidden = hasFrame;
    $('feed').parentElement.classList.toggle('stale', !node.fresh);
    if (hasFrame && node.fresh) {
      $('feed').src = '/api/frame?node=' + encodeURIComponent(node.id) + '&v=' + node.seq;
    } else if (hasFrame && $('feed').dataset.node !== node.id) {
      $('feed').src = '/api/frame?node=' + encodeURIComponent(node.id);
    }
    $('feed').dataset.node = node.id;
    $('empty-title').textContent = node.source === 'none' ? 'Choose a connected camera station' : 'Waiting for camera';
    $('empty-message').textContent = node.detail;
    let status = modes[node.source];
    if (!node.online) status = 'STATION OFFLINE';
    else if (!node.fresh) status = node.status === 'error' ? 'CAMERA ERROR' : 'WAITING FOR FRAMES';
    else if (node.detections.length) status += ' · POSSIBLE TURTLE';
    $('feed-status').textContent = status;
    $('feed-status').className = 'feed-status' + (isReplay ? ' replay' : node.detections.length ? ' alert' : '');
  }
  const events = [...data.events].reverse();
  const eventsKey = events.map(e => e.id).join('|');
  if (eventSignature !== eventsKey) {
    $('events').replaceChildren();
    for (const event of events) {
      const row = item('div', null, 'event');
      row.tabIndex = 0;
      row.setAttribute('role', 'button');
      row.setAttribute('aria-label', 'View ' + event.name);
      row.append(item('strong', 'Possible turtle'), item('span', Math.round(event.confidence * 100) + '%', 'score'),
        item('span', event.name, 'station'),
        item('time', new Date(event.at * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second:'2-digit'})));
      if (event.source === 'replay') row.append(item('span', 'Recorded footage', 'mode'));
      row.onclick = () => chooseStation(event.node_id);
      row.onkeydown = e => {if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); chooseStation(event.node_id);}};
      $('events').append(row);
    }
    eventSignature = eventsKey;
  }
  $('event-count').textContent = events.length;
  const peersKey = JSON.stringify(data.peers);
  if (peersKey !== peerSignature) {
    $('peers').replaceChildren();
    for (const peer of data.peers) {
      const row = item('li', null, 'peer');
      const label = item('span', peer.url, 'peer-label');
      label.title = peer.detail;
      const health = item('span', peer.status, peer.status === 'connected' ? 'healthy' : 'unhealthy');
      const remove = item('button', 'Remove');
      remove.onclick = async () => {
        try {await post('/api/peers/remove', {url:peer.url});}
        catch (error) {toast(error.message);}
      };
      row.append(label, health, remove);
      $('peers').append(row);
    }
    peerSignature = peersKey;
  }
}
async function poll() {
  try {
    const response = await fetch('/api/state', {signal: AbortSignal.timeout(4000)});
    if (!response.ok) throw new Error('Dashboard unavailable');
    render(await response.json());
  } catch (_) {
    $('connection').textContent = 'Disconnected · reconnecting';
    $('connection').className = 'connection offline';
    $('feed-status').textContent = 'CONNECTION LOST';
    $('feed').parentElement.classList.add('stale');
    $('in-view').textContent = '—';
    $('online').textContent = '—';
    $('latency').textContent = '—';
  } finally {
    setTimeout(poll, 700);
  }
}
$('station').onchange = e => chooseStation(e.target.value);
$('source').onchange = async () => {
  updatingControls = true;
  try {await post('/api/control', {source:$('source').value}); selected = current.local_id; stationChosen = $('source').value !== 'none';}
  catch (error) {toast(error.message);}
  finally {updatingControls = false;}
};
$('confidence').oninput = () => {$('confidence-value').textContent = $('confidence').value + '%';};
$('confidence').onchange = async () => {
  updatingControls = true;
  try {await post('/api/control', {confidence:Number($('confidence').value) / 100});}
  catch (error) {toast(error.message);}
  finally {updatingControls = false;}
};
$('peer-form').onsubmit = async e => {
  e.preventDefault();
  const button = e.target.querySelector('button');
  button.disabled = true;
  try {
    await post('/api/peers', {url:$('peer-url').value});
    $('peer-feedback').textContent = 'Connecting. Observations will appear under Camera view.';
    $('peer-url').value = '';
  } catch (error) {$('peer-feedback').textContent = error.message;}
  finally {button.disabled = false;}
};
$('copy').onclick = async () => {
  const field = $('address');
  try {
    if (!navigator.clipboard) throw new Error('Clipboard unavailable');
    await navigator.clipboard.writeText(field.value);
    toast('Dashboard link copied');
  } catch (_) {
    field.select();
    if (document.execCommand('copy')) toast('Dashboard link copied');
    else toast('Select and copy the network address');
  }
};
poll();
