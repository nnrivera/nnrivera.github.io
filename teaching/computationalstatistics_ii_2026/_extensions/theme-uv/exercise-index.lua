-- Run after Quarto renders theorem titles, so its numbering remains authoritative.
-- Enable in document/project YAML with: exercise-list: true

local function exercise_title(div)
  local title = nil
  div:walk({
    traverse = 'topdown',
    Span = function(span)
      if title == nil and span.classes:includes('theorem-title') then
        title = span.content:clone()
      end
    end
  })
  if title == nil then return nil end

  -- Remove the caption's bold wrapper and any links before making our own link.
  local plain = pandoc.Span(title):walk({
    Strong = function(el) return el.content end,
    Link = function(el) return el.content end
  }).content

  -- Quarto emits "Exercise X (name)". The list uses "Exercise X: name".
  if #plain > 0 and plain[#plain].t == 'Str' and plain[#plain].text == ')' then
    for i, inline in ipairs(plain) do
      if inline.t == 'Str' and inline.text == '(' then
        plain:remove(#plain)
        plain[i] = pandoc.Space()
        if i > 1 and plain[i - 1].t == 'Space' then
          plain[i - 1] = pandoc.Str(':')
        else
          plain:insert(i, pandoc.Str(':'))
        end
        break
      end
    end
  end
  return plain
end

function Pandoc(doc)
  if doc.meta['exercise-list'] ~= true or not quarto.doc.is_format('revealjs') then
    return nil
  end

  local slide_level = PANDOC_WRITER_OPTIONS.slide_level or 2
  local slide_id = nil
  local items = pandoc.List()
  local used_ids = {}
  doc:walk({
    Header = function(el) used_ids[el.identifier] = true end,
    Div = function(el) used_ids[el.identifier] = true end
  })

  for _, block in ipairs(doc.blocks) do
    if block.t == 'Header' and block.level <= slide_level then
      slide_id = block.identifier
    elseif block.t == 'HorizontalRule' then
      slide_id = nil
    else
      pandoc.Div({block}):walk({
        traverse = 'topdown',
        Div = function(div)
          if div.identifier:match('^exr%-') and div.classes:includes('exercise') then
            local title = exercise_title(div)
            if title ~= nil then
              if slide_id == nil or slide_id == '' then
                error('exercise-list: exercise ' .. div.identifier ..
                  ' needs a slide heading (for example ## Practice).')
              end
              items:insert({pandoc.Plain({pandoc.Link(title, '#' .. slide_id)})})
            end
          end
        end
      })
    end
  end

  -- No empty closing slide when there are no numbered exercises.
  if #items == 0 then return nil end

  local index_id = 'exercise-list'
  local suffix = 1
  while used_ids[index_id] do
    suffix = suffix + 1
    index_id = 'exercise-list-' .. suffix
  end
  local section_id = 'exercise-list-section'
  suffix = 1
  while used_ids[section_id] or section_id == index_id do
    suffix = suffix + 1
    section_id = 'exercise-list-section-' .. suffix
  end
  doc.blocks:insert(pandoc.Header(1, 'Exercise list',
    pandoc.Attr(section_id, {'unnumbered'})))
  doc.blocks:insert(pandoc.Header(slide_level, 'Exercises',
    pandoc.Attr(index_id, {'scrollable', 'unnumbered'})))
  doc.blocks:insert(pandoc.Div({pandoc.BulletList(items)},
    pandoc.Attr('', {'exercise-index', 'nonincremental'})))
  return doc
end
