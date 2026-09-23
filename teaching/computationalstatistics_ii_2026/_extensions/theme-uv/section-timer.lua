-- Section allocations are minutes; only level-one section dividers own timers.
function Header(el)
  local value = el.attributes['section-time']
  if not value then return nil end
  local minutes = tonumber(value)
  if el.level ~= 1 or not minutes or minutes <= 0 or minutes == math.huge then
    error('section-time: use a positive number of minutes on a level-one heading, e.g. # Topic {section-time="20"}.')
  end
  el.attributes['section-time'] = nil
  el.attributes['data-section-time'] = tostring(minutes)
  return el
end
