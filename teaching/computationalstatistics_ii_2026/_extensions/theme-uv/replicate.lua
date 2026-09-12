-- A replica is a destination, independent of explanations or slide layout.
function Div(el)
  if not el.classes:includes('replicate') then return nil end
  local ref = (el.attributes['ref'] or ''):gsub('^#', '')
  if ref == '' or ref:match('%s') then
    error('replicate: supply a label with ref="label", for example ref="thm-descent".')
  end
  if #el.content > 0 then
    error('replicate: leave the block empty; put commentary outside it.')
  end
  el.attributes['ref'] = nil
  el.attributes['data-replicate-ref'] = ref
  el.content = {pandoc.Para({pandoc.Str('Loading replica: ' .. ref)})}
  return el
end
