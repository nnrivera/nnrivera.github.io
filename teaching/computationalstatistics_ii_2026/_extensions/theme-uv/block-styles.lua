-- Numbered palettes apply before the other filters transform custom blocks.
function Div(el)
  local style = el.attributes['block-style']
  if not style then return nil end
  if not style:match('^[1-9]$') and style ~= '10' then
    error('block-style: choose an integer from 1 to 10, for example block-style="1".')
  end
  el.classes:insert('block-styled')
  el.classes:insert('block-style-' .. style)
  el.attributes['block-style'] = nil
  return el
end
