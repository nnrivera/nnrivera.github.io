-- Keep explanation content in Markdown so Quarto can render its mathematics.
function Div(el)
  if not el.classes:includes('explanation') then return nil end
  if el.attributes['ref'] then
    error('explanation: put an empty ::: {.replicate ref="label"} block inside the explanation instead of using ref on the explanation itself.')
  end

  local title = el.attributes['title'] or 'Explanation'
  if not quarto.doc.is_format('html') then
    el.content:insert(1, pandoc.Para({pandoc.Strong({pandoc.Str(title)})}))
    return el
  end

  local function escape(text)
    return text:gsub('&', '&amp;'):gsub('<', '&lt;'):gsub('>', '&gt;')
      :gsub('"', '&quot;')
  end
  local id = el.identifier ~= '' and (' id="' .. escape(el.identifier) .. '"') or ''
  local classes = escape(table.concat(el.classes, ' '))
  local result = pandoc.List({pandoc.RawBlock('html',
    '<details class="' .. classes .. '"' .. id .. '><summary>' .. escape(title) ..
    '</summary><div class="explanation-content">')})
  result:extend(el.content)
  result:insert(pandoc.RawBlock('html', '</div></details>'))
  return result
end
